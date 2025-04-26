
# Speech Emotion Recognition - Hướng dẫn cài đặt và chạy

## 1. Cài đặt thư viện

Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết:

```bash
pip install -r requirements.txt
```
## 2. Tải trọng số (model weight)

- Truy cập link sau:  [Kaggle - Speech Emotion Recognition (97.25% Accuracy)](https://www.kaggle.com/code/mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy#Saving-Best-Model)
- Tải về toàn bộ các file trọng số (weights) từ phần Output của notebook.
- Tạo thư mục model_weight tại thư mục gốc của dự án.
- Đặt tất cả các file tải về vào bên trong thư mục model_weight.

## 3. Chạy mô hình

Chạy lệnh sau để thực thi mô hình:
```bash
python main.py
```
