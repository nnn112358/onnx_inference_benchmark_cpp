# onnx_inference_benchmark_cpp

# ONNX推論ベンチマーク

ONNX Runtimeを使用してONNXモデルの推論性能をベンチマークするためのC++ユーティリティ

## 概要

このツールは、ONNXモデルの推論速度を複数の反復にわたって測定する簡単な方法を提供します。モデルの入力形状に基づいてランダムな入力データを自動生成し、指定された回数の反復を実行して、詳細なパフォーマンス統計を報告します。

## 特徴

- ベンチマーク用にあらゆるONNXモデルを読み込んで実行可能
- モデル入力情報（名前、次元、タイプ）の表示
- 適切なランダム入力データの自動生成
- ベンチマーク前のウォームアップ実行
- 包括的なパフォーマンス指標の報告：
  - 平均推論時間
  - 中央値推論時間
  - 最小・最大時間
  - 標準偏差
  - スループット（1秒あたりの推論回数）

## 要件

- C++17互換コンパイラ
- ONNX Runtimeライブラリ（バージョン1.21.0でテスト済み）

## インストール方法

1. [ONNX Runtimeのリリースページ](https://github.com/microsoft/onnxruntime/releases)からプリビルドパッケージをダウンロードしてONNX Runtimeをインストール

2. MakefileをONNX Runtimeのインストールパスで更新：
   ```
   ONNXRUNTIME_ROOT = /path/to/onnxruntime/
   ONNXRUNTIME_LIB = /path/to/onnxruntime/lib/
   ```

3. ベンチマークツールのビルド：
   ```
   make
   ```

## 使用方法

基本的な使い方：
```
./onnx_inference_benchmark <モデルへのパス> [反復回数]
```

例：
```
# 10回の反復を実行（デフォルト）
./onnx_inference_benchmark model.onnx

# 100回の反復を実行
./onnx_inference_benchmark model.onnx 100

# Makefileターゲットを使用（model.onnxで100回の反復を実行）
make run
```

## 出力例

```
Loading ONNX model: model.onnx
Number of inputs: 1
Input 0 name: input
Input 0 dimensions: 1 3 224 224
Input 0 type: 1
Generating random input data of size 150528
Running 100 iterations...
Completed 10 iterations
Completed 20 iterations
...
Completed 100 iterations

===== Inference Performance Results =====
Model: model.onnx
Iterations: 100
Total time: 5235.218 ms
Average time: 52.352 ms
Median time: 51.891 ms
Min time: 48.763 ms
Max time: 58.942 ms
Standard deviation: 2.105 ms
Throughput: 19.102 inferences/second
```

## プロジェクト構造

- `onnx_inference_benchmark.cpp` - ベンチマーク実装を含むメインソースファイル
- `Makefile` - ビルドシステム設定

## ビルドターゲット

- `make`または`make all` - ベンチマークプログラムをビルド
- `make run` - デフォルトパラメータでベンチマークを実行
- `make clean` - オブジェクトファイルと実行ファイルを削除
- `make help` - ヘルプ情報を表示

## 高度な使用法

ベンチマークプロセスをカスタマイズするためにソースコードを変更できます：

- `session_options.SetIntraOpNumThreads(1)`でスレッド数を変更
- `session_options.SetGraphOptimizationLevel()`で最適化レベルを調整
- 入力データのカスタム前処理の追加
- より詳細なプロファイリング測定の追加
