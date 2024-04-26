# Text Classification

## Cây thư mục
```
├── model/
│   ├── data/*
│   ├── task_2-3-4/ (Bài 2.3, 2.4)
│   │   ├── knn_svm.ipynb (Sử dụng mô hình KNN, SVM; bao gồm cả training - Bài 2.3 và đánh giá - Bài 2.4)
│   │   ├── simple_bert_trainer.py (Sử dụng pretrained model BERT - Bài 2.3)
│   │   ├── two_stage_bert_trainer.py (Sử dụng pretrained model BERT kết hợp triplet loss - Bài 2.3)
│   │   ├── knn_svm.ipynb (Huấn luyện và đánh giá mô hình KNN, SVM - Bài 2.3 và Bài 2.4)
│   │   ├── eval_bert.ipynb (Đánh giá mô hình BERT - Bài 2.4)
│   │   └── ...
│   ├── task_1.py (Bài 1)
│   ├── task_2-1_preprocess.ipynb (Bài 2.1 - Chuẩn bị dữ liệu)
│   ├── task_2-2_explore.ipynb (Bài 2.2 - Khám phá dữ liệu)
│   └── ...
│
├── triton-server/ (Bài 2.5)
│   ├── client/*
│   ├── model_repository/*
│
├── fastapi/ (Bài 2.5)
│   ├── app.py
│   ├── triton_client.py
│   └── ...
├── README.md
└── ...
```
## Huấn luyện mô hình
```bash
cd ./model
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
* KNN, SVM: run all block trong file `./task_2-3-4/knn-svm.ipynb`
* BERT:
```bash
cd ./task_2-3-4
python simple_bert_trainer.py
python two_stage_bert_trainer.py
```
## Triển khai thành API

### Triển khai mô hình lên Triton Server
```bash
cd ./triton-server
docker run --gpus=all -it --shm-size=256m --rm -p8000:8000 -p8001:8001 -p8002:8002 -v ${pwd}/model_repository:/models anhnh2002/tritonserver:v1
cd ..
cd ..
tritonserver --model-repository=/models
```

### Triển khai API
```bash
cd ./fastapi
uvicorn app:app --reload --port 2005
```
