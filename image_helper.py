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
        [在 batch_size > 1 時不用]
        """
        try:
            image = Image.open(self.image_path).convert("RGB") # 1. 載入圖片

            # 2. 準備輸入
            # 這份 instruciton 暫時不會用到 for 模型大小太小 跑不出來
            prompt_instruction = f"""你是一位服飾商品企劃與屬性標註員。系統會提供一張單品圖片。
任務：請只輸出一個可被 JSON.parse 解析的物件，描述該單品的關鍵屬性。不得加入任何額外文字、註解或反引號。
原則：

僅根據可見資訊作答；看不清楚或無法確定時填入 "unknown"。

嚴格遵守下列鍵名與可選值（受控詞彙）。若不在列表中，回 "unknown"。

請一併回傳每個欄位的 confidence（0–1，保留兩位小數），代表你對該欄位判斷的信心。

輸出格式（JSON 物件，單一物件、可被 JSON.parse 解析）：
{{
"類別": {{"value": "...", "confidence": 0.00}},
"版型": {{"value": "...", "confidence": 0.00}},
"領型": {{"value": "...", "confidence": 0.00}},
"袖長": {{"value": "...", "confidence": 0.00}},
"長度": {{"value": "...", "confidence": 0.00}},
"材質": {{"value": "...", "confidence": 0.00}},
"圖案": {{"value": "...", "confidence": 0.00}},
"細節": {{"value": ["..."], "confidence": 0.00}},
"風格": {{"value": ["..."], "confidence": 0.00}},
"場景": {{"value": ["..."], "confidence": 0.00}}
}}

可選值（受控詞彙）：

類別：上衣｜襯衫｜針織衫｜T恤｜外套｜洋裝｜裙裝｜褲子｜套裝｜鞋｜包｜配件｜unknown

版型：合身｜標準｜寬鬆｜落肩｜A字｜H直筒｜修身｜unknown

領型：圓領｜V領｜高領｜翻領｜連帽｜立領｜一字領｜unknown

袖長：無袖｜短袖｜七分袖｜長袖｜落肩長袖｜unknown

長度（上衣/外套）：短版｜常規｜中長｜長版｜unknown；（下身/洋裝/裙）：迷你｜及膝｜中長｜及踝｜拖地｜unknown

材質：棉｜麻｜絲｜羊毛｜聚酯纖維｜尼龍｜牛仔｜皮革｜針織｜混紡｜蕾絲｜雪紡｜unknown

圖案：純色｜條紋｜格紋｜波點｜印花｜字母｜Logo｜迷彩｜拼接｜提花｜unknown

細節（多選）：口袋｜拉鍊｜鈕扣｜綁帶｜荷葉邊｜開衩｜刺繡｜貼布｜抽繩｜鬆緊腰｜unknown

風格（多選）：極簡｜通勤｜街頭｜運動｜復古｜甜美｜學院｜雅痞｜機能｜度假｜正式｜unknown

場景（多選）：日常｜通勤｜約會｜派對｜運動｜居家｜旅行｜正式場合｜unknown

請僅輸出上述 JSON 物件；若任一欄位無法確定，將該欄位的 value 設為 "unknown" 並合理給出較低的 confidence。"""
            
            final_llava_prompt = "USER: <image>\n請詳細描述這件衣服的類別、樣式、圖案和細節，以 json format 回傳。 ASSISTANT:"
            # 3. 準備輸入 (注意：已移除 .to(device))
            # device_map="auto" 會自動處理設備分配
            inputs = processor(images=image, text=final_llava_prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 4. 取得輸入 token 的長度，以便稍後分離
            input_ids = inputs["input_ids"]
            input_length = input_ids.shape[1]

            # 5. 生成文字 (token IDs)
            with torch.no_grad():
                # 傳入 inputs 字典，並設定最大新 token 數量
                output_ids = model.generate(
                    **inputs, 
                    max_new_tokens=150,  # 設定一個合理的上限，例如 150
                    do_sample=False      # 使用 greedy decoding 獲取最可能的描述
                )

            # 6. 只解碼新生成的 token
            # output_ids 包含 prompt + answer，我們只取 answer
            # output_ids[0] 是因為 batch size 為 1
            new_tokens = output_ids[0, input_length:]
            
            # 7. 將 token IDs 解碼回文字
            caption = processor.decode(new_tokens, skip_special_tokens=True)
            
            # 8. 儲存
            self.text_description = caption.strip()

        except FileNotFoundError:
            print(f"Caption 錯誤: 圖片未找到 {self.image_path}。")
            self.text_description = "Image not found"
        except Exception as e:
            print(f"Caption 錯誤 ({self.image_path}): {e}")
            self.text_description = "Error generating caption"