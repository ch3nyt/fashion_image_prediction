# predict: given season, category, ... and return suitable related image(consider sales)
import os, pandas as pd, pickle, tqdm, torch
from image_helper import image_info # 仍然需要 import class 定義
import argparse
from transformers import CLIPModel, CLIPProcessor


def load_data(args):

    # 檢查快取檔案是否存在
    if os.path.exists(args.save_path):
        with open(args.save_path, "rb") as f:
            # 'rb' = Read Binary (二進位讀取)
            image_cat = pickle.load(f)
    else:
        raise LookupError
    
    # --- 在這裡，不論是載入的還是新產生的，image_cat 都已準備就緒 ---
    return image_cat

def count_distance():
    return


def train_decoder(image_cat):
    # train decoder through (image, text_description) pairs in image_cat
    
    decoder = None
    return decoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='visuelle2/')
    parser.add_argument("--image_root", type=str, default="images/")
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--save_path", type=str, default="image_info_whole.pkl")
    # 加上 targeted category, color, fabric, season
    # 現在先隨便試 + 測試
    args = parser.parse_args()

    image_cat = load_data(args)  # dictionary of image_info
    print()
    # 分 season，用前 11 個禮拜當作訓練資料，預測最後 1 週的銷量
    '''
    沒有 targeted 特徵
        - 預測銷量
            。Naive: 取前 11 週銷量平均當作 12 週銷量
            。其他: 拿 visuelle 裡面 foreso 的方法
        - 取各 category 用銷量加權 emb. -> 得到加權平均 emb.
        - 回傳成一段 text 描述
        - 用以生成圖片 (與該 category 真正最熱銷的商品做比較)
    
    有 targeted 特徵
        - 算 text-image 距離

    '''

    # 單純以向量加權不可行: 沒有意義 => 需要以語意去看
    '''
    # category 以 shorts 為例
    total_sales, total_emb = 0, 0
    best_sales, best_image_external_code = -1, -1 
    for _, info_object in tqdm(image_cat.items()):
        if info_object.category == "shorts":
            pred_sales = sum(info_object.sales_trend[:-1])/11  # naive prediction of sales
            total_sales += pred_sales
            total_emb += info_object.emb * pred_sales
            if pred_sales > best_sales:
                best_sales = pred_sales
                best_image_external_code = info_object.external_code
            
    avg_emb = total_emb/total_sales
    # model turn avg_emb into text description
    '''

