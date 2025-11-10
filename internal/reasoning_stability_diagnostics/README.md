# 推論の安定性を診断するスクリプト

このディレクトリには，推論型モデルの推論の安定性を診断するスクリプト `calculate_reasoning_failure_stats.py` （以下，本スクリプト）が格納されています．  
swallow-evaluation-instruct（以下，SWE）が生成した評価詳細（参考：[評価結果の詳細を確認する](https://github.com/swallow-llm/swallow-evaluation-instruct-private/blob/main/TIPS.md)）を解析して，推論が完了していない応答の割合（以下，推論失敗率）および，推論が完了した応答のみで評価したスコア（以下，推論完了時スコア）を計測します．  

今後気が変わるかもしれませんが，本スクリプトは"一時しのぎ"として提供するつもりです．  
SWE自体に診断モードあるいは診断用の指標を組み込みたいのですが，やや手間がかかるので独立したスクリプトにしました．  

## 主な仕様

評価詳細の実態はParquet形式のファイルであり，ファイルの読み込み元は 1) ローカルストレージ および 2) HuggingFaceに自動アップロードしたもの（参考：[評価詳細の確認](https://github.com/swallow-llm/swallow-evaluation-instruct-private/blob/main/README_t4_a3_lcl.md)に対応しています．  
モデルIDおよびローカルストレージのパス，HFの場合はさらにorganizationを指定するだけで動きます．  

解析対象のベンチマークは，Qwen3-Swallow および GPT-OSS-Swallow で常時モニタリングしている日本語タスク・英語タスクぜんぶに対応しています．  
JMMLUやJEMHopQAなどの"言語別平均に含めない準用および任意のタスク"はデフォルトでは解析しません．  

診断に使う主な指標は，推論失敗率および推論完了時スコアです．  
推論失敗率を他のモデルと比べることで，推論の安定性を定量的に評価できます．  
推論完了時スコアを通常のスコアと比べることで，もしも推論を安定化させられた場合の性能の伸びしろを（楽観的に）推測できます．  
逆に言うと，推論の不安定性でスコアがどのくらい損なわれてるのかわかります．  

解析対象のモデルは Qwen3-Swallow および GPT-OSS-Swallow に対応しています．  
Qwen3-Swallowは推論タグ <think> を手がかりにして厳密に推論失敗を判定します．  
GPT-OSS-Swallowは推論タグがないので，文字列の長さと文字列繰り返しによるヒューリスティクスで推論失敗を大雑把に判定します．大雑把ですが推論の安定性を診断するには十分でしょう．  

## 環境構築

本スクリプトは Pipenv を使用した環境構築に対応しています．  
だれかuvに対応させてくれるとうれしいです．  

Pipenvで仮想環境を構築してアクティベートするコマンドは以下の通り．  

```bash
pip install pipenv

cd internal/reasoning_stability_diagnostics
pipenv install
# 仮想環境に入る
pipenv shell
```

唯一の注意点は datasets パッケージは 4.0.0 未満 を使用することです．  
3.xと4.xに互換性がないので動かなくなります．  

## 実行方法

本スクリプトは2つのモードで実行できます：

1. **ローカルストレージモード**: 評価詳細がローカルに保存されている場合
2. **HuggingFaceモード**: 評価詳細がHuggingFaceにアップロードされている場合

実行例が `run_calculate_reasoning_failure_stats_{local,huggingface}.sh` に用意されています．  

### ローカルストレージモードで実行

評価詳細がローカルに保存されている場合に使用します．  
要点は lighteval実行時引数 `--output-dir` を `--lighteval-output-dir` にセットすること，  
および model_id と provider を lighteval 実行時の設定と合わせることです．参考：[lighteval 実行時引数](https://github.com/swallow-llm/swallow-evaluation-instruct-private)  

```bash
cd internal/reasoning_stability_analysis
pipenv run python calculate_reasoning_failure_stats.py \
    --model_id "Qwen/Qwen3-8B/reasoning" \
    --lighteval-output-dir "{lighteval実行時引数のoutput-dir}" \
    --reasoning_starter "<think>" \
    --provider "hosted_vllm"
```

#### コマンドライン引数の説明

- `--reasoning_starter`: （必須）推論開始タグを指定します
  - 例: `--reasoning_starter "<think>"`
  - **gpt-oss系列は推論開始タグがないので `"None"` を指定してください**．Noneにすると繰り返しに基づく判定のみが使用されます
- `--repetition-ngram`: （オプション）N-gramのサイズ（デフォルト: 50）
  - 繰り返しを検出するための文字N-gramのサイズ
- `--top-ngram-freq-repetition-threshold`: （オプション）最頻N-gramの閾値（デフォルト: 10）
  - この値以上の頻度で同じN-gramが出現する場合，推論が完了していないと判定されます

例（gpt-oss系列の場合）:
```bash
pipenv run python calculate_reasoning_failure_stats.py \
    --model_id "tokyotech-llm/GPT-OSS-Swallow-v0.1-ablation-LR-3.0E-5-iter0012500" \
    --lighteval-output-dir "{lighteval実行時引数のoutput-dir}" \
    --reasoning_starter "None" \
    --provider "hosted_vllm"
```

### HuggingFaceモードで実行

評価詳細がHuggingFaceに自動アップロードされている場合に使用します．
ローカルストレージモードとの違いは `--hf_organization` を追加することだけです．  

```bash
cd internal/reasoning_stability_analysis
pipenv run python calculate_reasoning_failure_stats.py \
    --model_id "tokyotech-llm/Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000" \
    --hf_organization "tokyotech-llm" \
    --reasoning_starter "<think>" \
    --provider "hosted_vllm"
```

MMLUシリーズおよびAIMEは，HuggingFaceモードでは正しく動作しないかもしれません．  

## 実行結果

本スクリプトは解析結果を標準出力・CSV・JSONの3つの形式で出力します．  
CSV・JSONファイルの保存先は以下の規則で決まります：

- **HuggingFaceモード**: `results/{hf_organization}/details_{provider}__{model_id}_private.{csv|json}`
- **ローカルストレージモード**: `results/LOCAL/details_{provider}__{model_id}_private.{csv|json}`

例:
```
results/tokyotech-llm/details_hosted_vllm__tokyotech-llm__Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000_private.csv
results/tokyotech-llm/details_hosted_vllm__tokyotech-llm__Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000_private.json
```

コンソールには以下のような簡易的なCSV形式で表示されます：

```
tokyotech-llm/Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000
task_id,reasoning_failure_ratio,performance_in_completion,performance
swallow|japanese_mt_bench|0,0.059,0.866,0.831
swallow|mifeval_ja|0,0.285,0.691,0.512
swallow|jamcqa|0,0.162,0.466,0.431
...
```

### CSV出力の形式

CSVファイルには以下のカラムが含まれます：

| カラム名 | 説明 |
|---------|------|
| `model_id` | モデルID |
| `task_id` | lightevalタスクID．例： `swallow|jamcqa|0` |
| `num_responses` | 全応答数 |
| `num_non_closed_reasoning` | 推論が完了していない応答数 |
| `num_closed_reasoning` | 推論が完了した応答数 |
| `reasoning_failure_ratio` | 推論失敗率 |
| `performance_in_completion` | 推論完了時スコア |
| `performance` | 通常の，全応答によるスコア |

### JSON出力の形式

JSONファイルは以下の階層構造で保存されます：

```json
{
  "Qwen/Qwen3-8B": {
    "swallow|japanese_mt_bench": {
      "num_responses": 800,
      "num_non_closed_reasoning": 47,
      "num_closed_reasoning": 753,
      "reasoning_failure_ratio": 0.05875,
      "performance_in_completion": 0.8656042496679947,
      "performance": 0.8305
    },
    ...
  }
}
```

以上

---

# Appenix

## 指標の定義

本スクリプトが計測する指標の定義を説明します．  

### 推論失敗率

SWEの評価詳細に推論過程は保存されてなくて応答文しかないのですけれど，  
推論が閉じてない場合は推論過程そのものを応答文とみなすというSWEの仕様を活用して，以下のいずれかの条件を満たす場合を「推論が完了していない」と定義しています：

1. **推論開始タグによる判定**: モデルの応答文が推論開始タグ（`--reasoning_starter`で指定、例: `<think>`）から始まる場合
   - 推論開始タグがない（`--reasoning_starter "None"`）場合は、この判定はスキップされます
2. **N-gramベースの判定**: 応答文が2万文字以上かつ，応答文中で同じ文字N-gram（デフォルト: 50文字）が一定回数以上（デフォルト: 10回）繰り返される場合
   - 推論ループなどで同じパターンが繰り返される異常を検出します

推論失敗率は，推論が完了していない応答の数を全応答数で割ったものです．  
設問1件につき複数の応答をするベンチマークは1件ずつ調べます．たとえばJHumanEvalは164問で1問につき10応答なので，全応答数は1,640件です．  

### 推論完了時スコア

推論完了時スコア（`performance_in_completion`）は，推論が完了した応答のみで計算される評価スコアです．  
通常の評価スコア（`performance`）と比較することで，推論の安定化によってどの程度性能が向上する可能性があるかを（楽観的に）推測できます．  

わかりやすい例としてJamC-QAのような1問1答のaccuracyを測るベンチマークでは，  
（推論が完了 かつ 正解した応答）÷ 推論が完了した応答 が，推論完了時スコアです．  
MT-BenchやLiveCodeBenchのような複数応答の場合は推論が完了した応答だけでスコアを再計算しています．  

推論完了時スコアの評価指標は通常の評価スコアと同じです．質問応答や数学・科学はaccuracy，コード生成はPass@1，MT-BenchはPreferenceです．  

Corpus BLEUの再計算が面倒なので，WMT20は推論完了時スコアを計測していません．  

## 指標の解釈

推論失敗率 reasoning_failure_ratio が高いモデルは推論が不安定だといえます．    
推論失敗率はベンチマークによって水準にかなり差があり，Qwen3-Swallowの場合はGPQA, AIME, LiveCodeBench, M-IFEval-Ja, JamC-QAの数値が高めです．  
推論失敗率が高いベンチマークの特徴は2つで，ひとつは 難易度が高くて推論過程が長くなるもの，もうひとつは日本語の一部ベンチマークです．  
後者はよくわかんない．  

推論完了時スコアと通常スコアの差 performance_in_completion > performance は，推論の安定化による性能向上の余地を示唆します．  
ただし設問の難易度と推論の不安定性のあいだには相関がありそうなので，楽観的な推測です．  
当然ですが推論失敗率が低ければスコア差は小さいです．  

以下は実際の出力例です（ [Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000](https://huggingface.co/datasets/tokyotech-llm/details_hosted_vllm__tokyotech-llm__Qwen3-Swallow-8B-v0.1-SFT-exp11-LR1.5E-5-iter0023000_private) ）．  

| ベンチマーク | reasoning_failure_ratio | performance | performance_in_completion | 分析 |
|--------|------------------------|-------------|---------------------------|------|
| GPQA Diamond | 0.495 | 0.434 | 0.630 | 推論失敗率が高く，不安定になりやすい設問（指示文）が多い |
| MATH-500 | 0.052 | 0.918 | 0.966 | 推論は比較的安定．スコアの伸びしろは小さい |
| M-IFEval-Ja | 0.285 | 0.512 | 0.691 | 不安定になりやすい設問（指示文）が多い，安定化によるスコアの伸びしろが大きい |
