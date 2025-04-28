
# Speech Emotion Recognition - Hướng dẫn cài đặt và chạy

## 1. Cài đặt thư viện

Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết:

```sh
pip install -r requirements.txt
```

## 2. Tải trọng số (model weight)

Quy ước `$PROJECT_ROOT/` là thư mục gốc của dự án.

- **Cách 1:**
  - Tạo thư mục `$PROJECT_ROOT/model_weight`
  - Truy cập link sau: [Kaggle - Speech Emotion Recognition (97.25% Accuracy)](https://www.kaggle.com/code/mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy#Saving-Best-Model)
  - Tải về toàn bộ các file trọng số (weights) từ phần Output của notebook vào thư mục `$PROJECT_ROOT/model_weight` vừa tạo.

- **Cách 2:** Sử dụng [Kaggle CLI](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md):

    ```sh
    cd $PROJECT_ROOT
    kaggle kernels output mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy -p ./model_weight
    ```

## 3. Chạy mô hình

Chạy lệnh sau để thực thi mô hình:

```sh
python main.py
```
