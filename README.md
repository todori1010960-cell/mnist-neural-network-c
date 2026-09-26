# MNIST Neural Network in C

C言語だけで実装した、MNIST手書き数字分類用の6層ニューラルネットワークです。
`6NN.c`ファイルのニューラルネットワーク部分を中心に、MNISTのIDX読込みだけを小さな自作ローダーとして分離しています。

## 構成

```text
784 inputs
  -> Fully Connected (50)
  -> ReLU
  -> Fully Connected (100)
  -> ReLU
  -> Fully Connected (10)
  -> Softmax
```

## 実装内容

- 全結合層、ReLU、数値的に安定なSoftmax
- Softmax + 交差エントロピーの逆伝播
- ミニバッチSGD
- He一様初期化
- データシャッフル
- 学習済みパラメータの保存・読込み
- MNIST IDX形式の画像・ラベル読込み

## ファイル

- `src/neural_network.c`: 元の`6NN.c`からニューラルネットワーク本体と学習処理を分離したもの
- `src/main.c`: 学習・推論のコマンドライン処理
- `src/mnist_loader.c`: 自作したMNIST IDXローダー
- `include/neural_network.h`: ニューラルネットワークの公開関数
- `include/mnist_loader.h`: データセット構造体とローダーの宣言

## ビルド

GCCまたはClangとMakeが必要です。

```bash
make
```

## データの配置

MNISTのIDXファイルを取得し、次のように配置します。データそのものはリポジトリに含めていません。

```text
data/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

## 学習

```bash
./mnist_nn train data fc1.dat fc2.dat fc3.dat
```

20エポックのミニバッチSGDを実行し、各エポックの損失と正解率を表示します。
学習後の重みは`fc1.dat`、`fc2.dat`、`fc3.dat`に保存されます。

## 推論

テストデータの画像番号を指定して推論します。

```bash
./mnist_nn infer data fc1.dat fc2.dat fc3.dat 0
```

指定した画像の各クラスの確率、予測結果、正解ラベルを表示します。

## 動作確認結果

MNISTの実データを用いて20エポックの学習と推論を確認しています。

- Train samples: 60,000
- Test samples: 10,000
- Network: 784 → 50 → ReLU → 100 → ReLU → 10 → Softmax
- Epochs: 20
- Batch size: 100
- Learning rate: 0.01
- Final train loss: 0.136491
- Final train accuracy: 96.10%
- Final test loss: 0.146428
- Final test accuracy: 95.62%