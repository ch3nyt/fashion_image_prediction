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
    llava_model = LlavaForConditionalGeneration.from_pretrained(llava_model_name).to(device)
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

    # --- 3. 產生 Embeddings ---
    test_count = 0
    for _, info_object in tqdm(image_cat.items()):
        info_object.get_text_description(llava_model, llava_processor, device)
        if test_count < 5:
            test_count += 1
            print(info_object.text_description)
        # info_object.get_embeddings(clip_model, clip_processor, device)

    with open(args.save_path, "wb") as f:
        # 'wb' = Write Binary (二進位寫入)
        # pickle.dump(您要儲存的物件, 檔案)
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