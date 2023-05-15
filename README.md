# Vision Transformer

基本的なVision Transformerです。

## Setup
初めにこのレポジトリをクローンしてください
```sh
git clone https://github.com/yabashikazuya/VisionTransformer.git
```

dockerのbuildを行います
```sh
cd VisionTransformer/docker
./build.sh
```

## データセット
データを以下のように配置してください
(例)  
├── dataset_train（学習データ）  
│   ├── dog  
│   │   ├── dog_train_0001.png  
│   │   ├── dog_train_0002.png  
│   │   ├── ：  
│   ├── cat  
│   │   ├── cat_train_0001.png  
│   │   ├── cat_train_0002.png  
│   │   ├── ：  
├── dataset_val（確認データ）  
│   ├── dog  
│   │   ├── dog_val_0001.png  
│   │   ├── dog_val_0002.png  
│   │   ├── ：  
│   ├── cat  
│   │   ├── cat_val_0001.png  
│   │   ├── cat_val_0002.png  
│   │   ├── ：  
├── dataset_test（テストデータ）  
│   ├── dog  
│   │   ├── dog_test_0001.png  
│   │   ├── dog_test_0002.png  
│   │   ├── ：  
│   ├── cat  
│   │   ├── cat_test_0001.png  
│   │   ├── cat_test_0002.png  
│   │   ├── ：  


## 実行方法
1. dockerに入ります
```sh
cd ~/VisionTransformer/docker
./run.sh
```
2. 以下を実行することで学習を行うことができます
```sh
./train.py
```

## AlexNet(CNN)との比較
以下のコードを用いることでAlexNetで学習を行うこともできます。
```sh
./Alexnet.py
```
ただし、Seed値の固定機能は実装予定のため対等な比較ではないことに注意してください。