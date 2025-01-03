# FER ResNet18
Author: 111550035

## Set up

1. 將 `111550035_HW4/` 資料夾上傳至 Google Drive。  
2. 將資料集 `data/` 放置於 `111550035_HW4/` 目錄下。  
3. 下載提供的權重檔案（`111550035_weight.txt`）並將資料夾 (`model/`) 放置於 `111550035_HW4/` 目錄下。  
4. 結構如下：  
    ```
    ├── 111550035_HW4.pdf  
    ├── 111550035_weight.txt  
    ├── README.md  
    ├── data  
    │   └── ... (Download from Kaggle)  
    ├── inference.ipynb  
    ├── model  
    │   ├── ResNet18.pth  
    │   ├── v0.pth  
    │   ├── v1.pth  
    │   ├── v2.pth  
    │   └── v3.pth  
    ├── resnet.py  
    ├── test.py  
    ├── train_ResNet18.py  
    ├── train_ResNet50.py  
    ├── utils.py  
    └── weights
        └── (Your own models)
    ```  
--- 

這樣每行都會獨立換行，目錄結構也會清楚顯示。如果有其他格式需求，請隨時告訴我！

## Testing

1. 打開 `inference.ipynd`
2. Run every cell in the first section
3. Run `!python test.py`
4. The output will be saved as `111550035_HW4/output.csv`

## Training

1. 打開 `inference.ipynd`
2. Run every cell in the first section
3. Run `!python train_ResNet18.py` or `!python train_ResNet50.py`
4. Your modes of every epoch will be saved to `11155035_HW4/weights/`
5. Choose your models and put them under `111550035_HW4/model/`
6. Edit the model path in `test.py` to your models' path
5. Start testing !!!
