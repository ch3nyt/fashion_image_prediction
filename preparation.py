import argparse
import pandas as pd
import torch
import os
from PIL import Image, ImageFile
from transformers import LlavaForConditionalGeneration, LlavaProcessor, CLIPModel, CLIPProcessor
import pandas as pd 
from tqdm import tqdm
from image_helper import image_info
import pickle 

ImageFile.LOAD_TRUNCATED_IMAGES = True  # ? # 允許載入被截斷的圖片

def run(args):

    # --- VLM (Vision-Language Model) ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading VLM model: {args.model_name}")
    llava_model_name = "llava-hf/llava-1.5-7b-hf"
    # ✅ 正確的 4-bit 載入程式碼 (修正版)
    llava_model = LlavaForConditionalGeneration.from_pretrained(
        llava_model_name,
        load_in_4bit=True,          # 關鍵：啟用 4-bit 量化
        torch_dtype=torch.float16,  # 雖然用 4-bit 載入，但計算仍用 16-bit
        device_map="auto"           # 自動將模型分配到 GPU
    )
    llava_processor = LlavaProcessor.from_pretrained(llava_model_name)
    llava_model.eval()

    clip_model = CLIPModel.from_pretrained(args.model_name).to(device)  # .to(device) 將模型權重移至 GPU (如果可用)
    clip_processor = CLIPProcessor.from_pretrained(args.model_name)  # 載入對應的處理器 (它會處理圖片的 resize, crop, normalize)
    clip_model.eval() 
    print(f"Model loaded, using device: {device}")

    sales = pd.read_csv(args.dataset_path + "sales.csv")    
    image_cat = {}
    for _, row in sales.iterrows():
        image_cat[row["external_code"]] = image_info(
            row["external_code"],
            args.dataset_path + args.image_root + row["image_path"],
            row["category"],
            row["color"],
            row["fabric"],
            row["season"],
        )
    
    # agg_sales: sum up sales from all retail stores of week 0-11 of different season.
    group_cols = ["external_code"]  # 沒有對齊 "release_date", Naive 地對時間序列做處理。另外 season 每個商品僅一個
    sales_cols = [str(i) for i in range(12)]
    agg_sales_subset = sales[group_cols + sales_cols]
    agg_sales = agg_sales_subset.groupby(group_cols).sum().reset_index()  # 僅 5577 筆資料(unique商品數)

    # sales record into dataset
    for _, row in agg_sales.iterrows():
        image_cat[row["external_code"]].sales_trend.extend([row[str(i)] for i in range(12)])

    # --- 3. 產生 Embeddings ---(batch=1 的時候)
    '''
    test_count = 0
    for _, info_object in tqdm(image_cat.items()):
        info_object.get_text_description(llava_model, llava_processor, device)
        if test_count < 5:
            test_count += 1
            print(info_object.text_description)
        # info_object.get_embeddings(clip_model, clip_processor, device)
    '''
    # --- 3. 產生 Embeddings (批次處理版) ---
    print("開始批次生成圖片描述...")

    # 依照你的要求設定 Batch Size
    BATCH_SIZE = 16

    # 將 image_cat 字典轉換為列表，方便批次切片
    info_objects_list = list(image_cat.values())
    
    # 使用 tqdm 顯示批次的進度 (i 會是 0, 16, 32, 48...)
    for i in tqdm(range(0, len(info_objects_list), BATCH_SIZE), desc="批次處理中"):
        
        # 1. 準備這一批次的資料
        batch_objects_slice = info_objects_list[i : i + BATCH_SIZE]
        
        image_batch = []
        prompt_batch = []
        valid_objects_in_batch = [] # 只儲存成功讀取的物件

        for info_object in batch_objects_slice:
            try:
                # 載入圖片
                image = Image.open(info_object.image_path).convert("RGB")
                
                # 準備 prompt (你可以換成你自己的 JSON prompt)
                prompt_text = "USER: <image>\n請詳細描述這件衣服的類別、樣式、圖案和細節，以 json format 回傳。 ASSISTANT:"
                
                # 將成功讀取的資料加入批次
                image_batch.append(image)
                prompt_batch.append(prompt_text)
                valid_objects_in_batch.append(info_object)
                
            except Exception as e:
                # 如果圖片損毀或遺失，則跳過
                print(f"警告：跳過無法讀取的圖片 {info_object.image_path}: {e}")
                info_object.text_description = "Error: Image file not found or corrupt."

        # 如果這整批圖片都讀取失敗，就跳到下一批
        if not valid_objects_in_batch:
            continue

        # 2. 批次處理 (Processor)
        #    Processor 會一次處理 BATCH_SIZE 數量的圖片和文字
        inputs = llava_processor(
            images=image_batch, 
            text=prompt_batch, 
            return_tensors="pt",
            padding=True  # 關鍵：將 prompts 補齊 (pad) 到同樣長度
        )
        
        # 3. 將整批 inputs 移到 GPU
        #    (這行程式碼適用於 float16 和 4-bit 載入)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        # 4. 批次生成 (Generate)
        #    模型會一次"看" BATCH_SIZE 張圖片，並"同時"生成 BATCH_SIZE 份答案
        with torch.no_grad():
            output_ids = llava_model.generate(
                **inputs,
                max_new_tokens=150, # 設定一個合理的生成上限
                do_sample=False
            )

        # 5. 批次解碼 (Decode) 並存回 object
        new_tokens = output_ids[:, input_length:] # [BATCH_SIZE, num_new_tokens]
        captions = llava_processor.batch_decode(new_tokens, skip_special_tokens=True)

        # 6. 將生成的 captions 存回對應的 info_object
        for info_object, caption in zip(valid_objects_in_batch, captions):
            info_object.text_description = caption.strip()
            
        # 為了模擬你原本的 test_count < 5，我們只印出第一批次的結果
        if i == 0:
            print("\n--- 第一批次生成結果 (範例) ---")
            for info_object in valid_objects_in_batch:
                print(f"  [{info_object.image_path.split('/')[-1]}] -> {info_object.text_description[:60]}...")
            print("----------------------------------\n")

    # (你的 get_embeddings 迴圈也應該用同樣的方式批次化)
    # ...

    with open(args.save_path, "wb") as f:
        pickle.dump(image_cat, f)

    print("儲存完畢。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='visuelle2/')
    parser.add_argument("--image_root", type=str, default="images/")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--save_path", type=str, default="image_info_whole.pkl")
    args = parser.parse_args()
    run(args)