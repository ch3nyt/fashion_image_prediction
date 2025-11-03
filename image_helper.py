import os
import torch
from PIL import Image, ImageFile
from transformers import CLIPModel, CLIPProcessor # 如果 get_embeddings 在這裡

ImageFile.LOAD_TRUNCATED_IMAGES = True

class image_info:
    def __init__(self, external_code, image_path, category, color, fabric, season):
        self.external_code = external_code
        self.image_path = image_path
        self.category = category
        self.color = color
        self.fabric = fabric
        self.season = season[:2]  # SS(Spring/Summer) 或 AW(Autumn/Winter)
        self.sales_trend = []
        self.text_description = ""
        self.emb = 0 # 初始 embedding
    
    def get_embeddings(self, model, processor, device):
        """
        使用預先載入的 VLM 模型和處理器來計算此圖片的 embedding。
        """
        try:
            
            image = Image.open(self.image_path).convert("RGB")
            
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                
            self.emb = image_features.cpu()
            
        except FileNotFoundError:
            print(f"Warning: 圖片未找到 {self.image_path}。 Embedding 設為 None。")
            self.emb = None
        except Exception as e:
            print(f"處理 {self.image_path} 時發生錯誤: {e}。 Embedding 設為 None。")
            self.emb = None
    
    def get_text_description(self, model, processor, device):
        """
        [新函式] 使用 BLIP 產生圖片描述
        """
        try:
            image = Image.open(self.image_path).convert("RGB") # 1. 載入圖片

            # 2. 準備輸入
            # 我們可以給 BLIP 一個 "prompt" 來引導它
            prompt = "a picture of" 
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

            # 3. 生成文字 (token IDs)
            with torch.no_grad():
                # .generate() 用於生成任務
                output_ids = model.generate(**inputs, max_length=50) # 可設定最大長度

            # 4. 將 token IDs 解碼回文字
            # [0] 是因為 batch size 為 1
            caption = processor.decode(output_ids[0], skip_special_tokens=True)
            
            # 5. 清理文字並儲存
            caption = caption.replace(prompt, "").strip() # 移除我們給的 prompt
            self.text_description = caption

        except FileNotFoundError:
            print(f"Caption 錯誤: 圖片未找到 {self.image_path}。")
            self.text_description = "Image not found"
        except Exception as e:
            print(f"Caption 錯誤 ({self.image_path}): {e}")
            self.text_description = "Error generating caption"