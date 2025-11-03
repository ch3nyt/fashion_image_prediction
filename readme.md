# Fashion Image Prediction 
(中文 readme)

## 介紹

我想要在給定`文字敘述、季節`下，去生成符合限制，且未來可能會熱銷的服飾圖案。
使用的資料集為`Visuelle 2.0`(https://github.com/HumaticsLAB/visuelle2.0-code)
```
/visuelle2
    |-sales.csv  # 商品在什麼店鋪、哪時候開賣、有哪些特徵標籤，以及開賣後 12 週每週銷量
    /images
        /AI17
            |-xxxxx.png
        /AI18
        /AI19
        /PE17
        /PE18
        /PE19
```
## 執行步驟

1. 將 `sales.df` 轉換成 `image_helper.py` 內自訂義 `image_info` 格式儲存成 `.pkl` 檔 
```
python preparation.py
```

2. 