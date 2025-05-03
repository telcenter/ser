
# Speech Emotion Recognition

- [Speech Emotion Recognition](#speech-emotion-recognition)
  - [Hướng dẫn cài đặt và chạy](#hướng-dẫn-cài-đặt-và-chạy)
    - [0. GPU](#0-gpu)
    - [1. Cài đặt thư viện](#1-cài-đặt-thư-viện)
    - [2. Tải trọng số (model weight)](#2-tải-trọng-số-model-weight)
    - [3. Chạy mô hình](#3-chạy-mô-hình)
  - [Confusion Matrix](#confusion-matrix)
  - [Credit](#credit)

## Hướng dẫn cài đặt và chạy

### 0. GPU

Trên Linux với NVIDIA GPU, để model có tốc độ chạy nhanh nhất,
cần cài đặt `cuda-toolkit` và `cudnn`.

Đối với Linux Mint 21.3 (hoặc Ubuntu 22.04), NVIDIA GPU (CUDA 12):

- Trước tiên làm theo [hướng dẫn tại đây](https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_local) để cài `cudnn`.
- Chạy lệnh sau để cài `cuda-toolkit`:

    ```sh
    sudo apt install cuda-toolkit
    ```

### 1. Cài đặt thư viện

Chạy lệnh sau để cài đặt tất cả các thư viện cần thiết:

```sh
pip install -r requirements.txt
```

### 2. Tải trọng số (model weight)

Quy ước `$PROJECT_ROOT/` là thư mục gốc của dự án.

- **Cách 1:**
  - Tạo thư mục `$PROJECT_ROOT/model_weight`
  - Truy cập link sau: [SER](https://www.kaggle.com/datasets/nvnhat04/ser-model)
  - Tải về toàn bộ các file trọng số (weights) từ phần Output của notebook vào thư mục `$PROJECT_ROOT/model_weight` vừa tạo.

- **Cách 2:** Sử dụng [Kaggle CLI](https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md):

    ```sh
    cd $PROJECT_ROOT
    kaggle kernels output mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy -p ./model_weight
    ```

### 3. Chạy mô hình

Chạy lệnh sau để thực thi mô hình:

```sh
python main.py
```

## Confusion Matrix

![confusion matrix](./docs/images/confusion-matrix.jpg)

## Credit

[The original notebook](https://www.kaggle.com/code/mostafaabdlhamed/speech-emotion-recognition-97-25-accuracy/notebook).
