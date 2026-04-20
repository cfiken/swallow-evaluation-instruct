# TSUBAME4，ACBI3.0，岡崎研HPC を用いた評価方法

> 目次
> - [概要](#概要)
> - [更新履歴](#更新履歴)
> - [1. 環境構築](#1-環境構築)
>   - [1.1 評価者固有情報の登録](#11-評価者固有情報の登録)
>   - [1.2 評価者固有情報ファイルの名前変更](#12-評価者固有情報ファイルの名前変更)
>   - [1.3 環境構築スクリプトの実行](#13-環境構築スクリプトの実行)
> - [2. 評価の実行](#2-評価の実行)
>   - [2.1 モデルの設定](#21-モデルの設定)
>   - [2.2 タスクの設定](#22-タスクの設定)
>   - [2.3 評価の実行](#23-評価の実行)
>   - [2.4 評価結果の確認](#24-評価結果の確認)
>   - [2.5 評価結果の確認](#25-評価結果の確認)
>   - [2.6 評価ログの確認](#26-評価ログの確認)
>   - [2.7 評価詳細の確認](#27-評価詳細の確認)
> - [3. Tips](#3-tips)
>   - [3.1 タスクを追加するときに](#31-タスクを追加するときに)
>   - [3.2 モデル固有の生成条件を追加するとき](#32-モデル固有の生成条件を追加するとき)
>   - [3.3 各モデル用のディレクトリについて](#33-各モデル用のディレクトリについて)
>   - [3.4 特定のproviderでエラーが出る場合の対応](#34-特定のproviderでエラーが出る場合の対応)
>   - [3.5 並列応答数を強制的に1にする方法](#35-並列応答数を強制的に1にする方法)
>   - [3.6 評価がいつまでたっても終わらない場合](#36-評価がいつまでたっても終わらない場合)
>   - [3.7 vLLMのreasoning_parserが対応していない推論型モデル](#37-vLLMのreasoning_parserが対応していない推論型モデル)
>   - [3.8 Baseモデルを無理やり評価する](#38-Baseモデルを無理やり評価する)
>   - [3.9 qsub_multiを使って複数モデルを効率的に評価する](#39-qsub_multiを使って複数モデルを効率的に評価する)

## 概要
この資料は，岡崎研の評価チームが TSUBAME4，ABCI3.0，岡崎研HPC 上で評価を行う際に参照することを想定した，内部向けのマニュアルである．

<details><summary>更新履歴</summary>

| 日付 | 内容 | 担当 |
| -- | -- | -- |
| 2025/06/19 | 初稿 | 齋藤 |
| 2025/07/01 | custom_settings とタスクの追加に関して追記 | 齋藤 |
| 2025/07/02 | Provider固有の問題への対処法、モデル固有の生成条件に関する注意、計算が終わらないときの対処法をTipsに追記 | 水木 |
| 2025/07/07 | custom_model_settings に関する追記・修正 | 齋藤 |
| 2025/07/25 | ABCI，local での実行に関する追記・修正 | 齋藤 |
| 2025/07/31 | 独自のreasoning parserを使うケースを追記 | 水木 |
| 2025/08/04 | Baseモデルを無理やり評価するユースケースを追記 | 水木 |
| 2025/09/27 | qsub_multiを使って複数モデルを効率的に評価する | 一瀬 |
| 2026/04/09 | 環境変数の説明の追記・修正，その他軽微な修正 | 松下 |

</details>

## 1. 環境構築
### 1.1 評価者固有情報の登録
`.env_template` に記されている以下の変数について各評価者の環境に合わせて修正を行う．

| カテゴリ | 変数名 | 説明 | 備考 | 
| -- | -- | -- | -- |
| Paths | `GROUP_AND_NAME` | 以下の変数の設定を簡単にするための変数． | 必ずしも使う必要はない． |
| Paths | `REPO_PATH` | 評価レポジトリの絶対パス．| デバッグなどで複数のレポジトリを持っている場合は注意． |
| Paths | `STORE_PATH` | 出力ファイルおよび評価集計ファイルの保存先絶対パス．| デバッグなどで複数のレポジトリを持っている場合は注意． |
| Paths | `HUGGINGFACE_CACHE` | HuggingFace のデータセットやモデルのインストールなどに関するキャッシュを保存しておくディレクトリ． | 必ず `/gs/bs/` 以下のディレクトリを指定すること． |
| Paths | `UV_CACHE` | UV のパッケージインストールなどに関するキャッシュを保存しておくディレクトリ．| 容量はそこまで大きくならないことが想定されるが，念の為 `/gs/bs/` 以下のディレクトリを指定するとよい．|
| Paths | `VLLM_CACHE` | VLLM の動作に関するキャッシュを保存しておくディレクトリ．| 容量がどれほど占められるかは把握していないが，念の為`/gs/bs/` 以下のディレクトリを指定するとよい．|
| Groups | `TSUBAME_GROUP` | 評価に用いる TSUBAME のグループ | デフォルトは岡崎研のものになっている． |
| Groups | `ABCI_GROUP` | 評価に用いる ABCI のグループ | デフォルトは産総研のものになっている． |
| APIs | `OPENAI_API_KEY` | MT-Bench の評価における LLM-as-a-judge で OpanAI のモデルを使用するときや，OpenAI のモデルを評価するときに使用する． | Swallow project 用の API を指定すること．お金を消費するので要注意．|
| APIs | `DEEPINFRA_API_KEY` | DeepInfra のモデルを評価するときに使用する． | Swallow project 用の API を指定すること．お金を消費するので要注意．|
| APIs | `HF_TOKEN` | HuggingFace のデータセットやモデルのインストール時に用いられるトークン．| Rate Limit の緩和や使用許可が必要なデータセット・モデルを使用する際に必要．|
| APIs | `OPENROUTER_API_KEY` | OpenRouter 経由でモデルを評価する時に使用する．| Swallow project 用の API を指定すること．お金を消費することがあるので要注意．|

### 1.2 評価者固有情報ファイルの名前変更
1.1 で評価者固有情報を記載したファイルの名前を`.env_template` から `.env` に変更しておく． \
これにより評価スクリプトから読み取られるようになり，また，`gitignore` の対象となる．
>[!Warning]
> `.env` には　API キーが含まれているため決してパブリックに公開してはならない．

### 1.3 環境構築スクリプトの実行
1.1，1.2 で正しく評価者固有情報が登録できていることを確認した上で，以下のスクリプトを実行し，環境構築を行う．
```bash
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruct-private
bash scripts/qsub/environment/setup_t4_uv_envs.sh
```

最終的に `"✅ Environment was successfully created!"` が表示されれば成功．


## 2. 評価の実行
### 2.1 モデルの設定
まず，評価を行うモデルについて以下の情報を`scripts/qsub/qsub_all.sh`の `# Set Args` の欄に書き込む．

| カテゴリ | 変数名 | 説明 | 備考 |
| -- | -- | -- | -- |
| Common | `NODE_KIND` | 使いたいノード．<br> tsubame ["node_q", "node_h", "node_f", "cpu_16"]，<br> abci ["rt_HG", "rt_HF", "rt_HC"]，<br> local ["gpu_*" (*はGPUの数), "cpu"] | 13B以下なら"node_q"や"rt_HG"や"gpu_2"，<br> 13B超なら"node_f"や"rt_HF"や"gpu_4"，<br> OpenAIやDeepInfraのAPIを使うなら"cpu_16"や"rt_HC"や"cpu"を選ぶと良い．|
| Common | `MODEL_NAME`| 評価するモデルの HuggingFaceID または絶対パス．| HuggingFaceID は HuggingFace のモデルカード上部にあるコピーボタンから取得できる．openAIのmodelに関しては、openai/~のようにすること.|
| Special | `PROVIDER` | 評価するモデルを serve するための provider．| HuggingFaceのモデルであれば"vllm"（デフォルト），OpenAI のモデルなら"openai"を指定．Deepinfra を使う場合は"deepinfra"を，OpenRouter を使う場合は "openrouter" 指定する．|
| Special | `CUSTOM_SETTINGS` | 使いたい custom model settings の名前．SYSTEM_MESSAGEなどモデル固有の設定を渡す事ができる．詳細は「3.2 モデル固有の生成条件を追加するとき」を参照．| 例：`default`，`reasoning`，`code`． |
| Special | `PREDOWNLOAD_MODEL` | ジョブを投げる前に HuggingFace からモデルをダウンロードするかどうかの設定．| OpenAI や DeepInfra のモデルを評価する場合は必ず `false` にすること． |
| Special | `MAX_SAMPLES` | 評価に用いるサンプル数の上限． | 主にデバッグの時に用いる．デフォルトは無制限．| 
| Special | `UPLOAD_DETAILS` | HuggingFace に評価の詳細（モデルの出力など）をアップロードするかの設定． | 基本は `false` で，評価依頼でアップロードを要請された際のみ `true` にすること． | 
| Special | `COPY_DETAILS` | abciでのみ設定可能．共有ディレクトリに評価の詳細（モデルの出力など）をコピーするかの設定 | 共有ディレクトリは `/groups/gag51395/share/se_eval_details`．| 
| Special | `AR_ID` | 予約したノードで評価を実行する際に指定する，予約ノードの AR ID． | 通常の評価の際は空欄で良い．| 
| Environmental | `SERVICE` | 評価に使うサービス（実行環境）| tsubame（TSUBAME4），abci（ABCI3.0），local（岡崎研HPC）に対応している． |
| Environmental | `PRIORITY` | 使いたいTSUBAME4における優先度．["-5", "-4", "-3"] | 数値が大きい方がジョブが流れやすくなる．しかし，それに応じて値段が2倍，4倍と高くなるので，指定する場合は要相談．|
| Environmental | `CUDA_VISIBLE_DEVICES` | local で使いたい GPU リソースの指定．| |

>[!Note]
> openaiやdeepinfraのようなAPIを使う場合はlocalでの実行を推奨する．

### 2.2 タスクの設定
2.1 でモデルの設定を終えたら \
同ファイル（`scripts/qsub/qsub_all.sh`）下部の `# Submit tasks` 以降を編集し， \
評価するタスクを指定する．具体的には，評価しないタスクについてコメントアウトをすれば良い．

### 2.3 評価の実行
2.1，2.2 でモデルとタスクの指定を正しく行ったことを確認したのち，以下のスクリプトで評価のジョブを投げることができる．
```sh
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruct-private
bash scripts/qsub/qsub_all.sh
```

### 2.4 評価状況の確認
以下のスクリプトから評価状況を確認することができる．
```sh
# 以下は Saito の例
cd /gs/fs/tga-okazaki/saito/swallow-evaluation-instruct-private
bash scripts/qsub/utils/save_and_check_qstat.sh
```

### 2.5 評価結果の確認
評価結果（`aggregated_results.json`）は \
各モデル用のディレクトリ（`{STORE_PATH}/results/{provider}/{model_publisher}/{model_name}/{custom_settings}`）以下に保存される． \
評価担当者は `aggregated_results.json` 内の `overall`の値を，指定された spreadsheet にコピーすれば良い．

ちなみに評価結果には評価メトリクスや評価タスクに加え，使用した custom_settings の詳細も記されている． \
もちろん custom_settings を使用していない場合には何も記されない．


### 2.6 評価ログの確認
評価のログ（標準出力 `.o`/`.OU` ファイル・標準エラー出力 `.e`/`.ER` ファイル）は \
各モデル用のディレクトリ（`{STORE_PATH}/results/{provider}/{model_publisher}/{model_name}/{custom_settings}`）以下に言語・タスクごとに保存される．


### 2.7 評価詳細の確認
評価結果の詳細は `lighteval/outputs/results` 以下に `.json` ファイルとして， \
モデルが生成した回答の詳細は `lighteval/outputs/outputs` 以下に `.pqt` ファイルとして保存されている． \
`.pqt`ファイルは pandas を用いて dataframe として開くことができる． \
（`scripts/utils/details_viewer.ipynb`参照）  
[評価結果の詳細](./TIPS.md)に，評価詳細 dataframe の項目について簡易な説明がある．  

これらのjsonファイルおよびpqtファイルは，HuggingFace Datasetsに自動アップロードして共有したりブラウザで閲覧（要有料プラン）することができる．
事前にHuggingFace CLIで認証したうえで，lightevalを以下の引数で実行すれば，jsonとpqtが自動的にアップロードされる [Saving and reading results](https://huggingface.co/docs/lighteval/v0.8.0/en/saving-and-reading-results)．  
なおアップロードしたDatasetはデフォルトでprivate設定なので，意図しない限りインターネットに公開されることはない．  

```
lighteval endpoint litellm \
    (中略)
    --save-details \
    --push-to-hub \
    --results-org "{あなたが属しているHuggingFace Organization ID}" # たとえば　tokyotech-llm
```

HuggingFaceに自動アップロードすると，モデルIDがデータセット名，タスクIDがデータセットのSubset，タイムスタンプがSplitになる．  
スコアなどが記録されるjsonファイルは，全タスクまとめて `results` という Subset に保存される．  
有料プランを契約している場合はDataset Viewerによってブラウザで中身を閲覧できる．無料プランの場合は一般公開にした場合のみDataset Viewerが有効になる．  

Qwen3-8B の評価詳細（MATH-500, GPQA, GPQA-Ja 各5問のみ）を自動アップロードしたので，閲覧してイメージをつかんでほしい．  
https://huggingface.co/datasets/s-mizuki-nlp/details_hosted_vllm__Qwen__Qwen3-8B_private

評価詳細を分析したいときは datasets パッケージでダウンロードするとよい．  

```
import datasets

details = datasets.load_dataset("データセットID", subset="タスクID", split="タイムスタンプ（任意）")
```

## 3. その他
### 3.1 タスクを追加するときに
タスクを追加する際には `lighteval/src/ligtheval/tasks/swallow/` 以下にタスクの定義を書く． \
しかし，その操作はあくまで lighteval に対するの操作であり，swallow-evaluation-instruct 用には追加の操作が必要である． \
以下にそれをまとめる．

| カテゴリ | 必要性 | 操作対象 | 操作内容 |
| -- | -- | -- | -- |
| 結果集約（Aggregate）のための操作 | 必須 | `scripts/aggregate_utils/conf.py` | 追加したタスクに対応するメトリクスを定義する| 
| | 適宜 |`scripts/aggregate_utils/funcs.py` | 追加したタスクのメトリクスに必要な計算を追加することができる |
| | 適宜 |`scripts/aggregate_utils/white_lists.py` | 追加したタスクのメトリクスの計算に用いるタスクサブセットのサブセットを定義することができる |
| 評価実行のための操作 | 必須 | `scripts/qsub/conf/tasks_runtime.csv` | 追加したタスクについて，`key`（"{言語}_{タスク名}"），`script`（定義したタスク名），`result_dir`（結果・ログの出力先），`framework`（フレームワーク），`hrt_q`（node_qでの想定所要時間），`hrt_f`（node_fでの想定所要時間），`wlt_HG`（rt_HGでの想定所要時間），`wlt_HF`（rt_HFでの想定所要時間），を定義する |
| | 必須 | `scripts/qsub/qsub_all.sh` | 追加したタスクについて，`qsub_task {言語} {タスク名}` を末尾の適当な箇所に追加する．|
| | 必須 | `scripts/generation_settings/task_settings.csv` | 追加したタスク固有の生成条件を記す．デフォルトは`temperature=0.0`．キーとバリューは"="で繋ぎ，複数の条件を定義したい場合は","で繋ぐ．|


### 3.2 モデル固有の生成条件を追加するとき
モデルによっては意図した推論をさせるために temperature や system message を指定する必要がある． \
それらの指定は `scripts/generation_settings/custom_model_settings` 以下に \
model publisher ごと，model name ごとに .yaml ファイルで定義することができる．

例えば，`tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3` についての指定を追加したい場合は \
`scripts/generation_settings/custom_model_settings/tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3.yaml` \
に定義を行えば良い

よく使われる設定に関しては`scripts/generation_settings/custom_model_settings/common_settings.yaml`に書いてありこれを使用することもできる

具体的な定義の例や定義できるパラメタについては　\
`scripts/generation_settings/custom_model_settings/template.yaml` に書かれている．\
必要に応じてこの `template.yaml` を複製するなどして活用してほしい．

>[!Warning]
> temperature を指定する custom_model_settings は，\
> デフォルトのtemperatureが0になっているベンチマークに対してのみ，適用すること． \
> MT-BenchやLiveCodeBenchのようにデフォルトのtemperatureが0でないベンチマークは，\
> 特別な要求がない限り，custom_model_settings を指定しないで実行すること．

>[!Note]
> 特定の推論が必要な場合，原則，評価依頼者が評価依頼時に生成条件を指定する． \
> しかし，場合によっては依頼者が勘違いや指定漏れをすることがありうる． \
> そのため以下のケースに該当するにもかかわらず生成条件の指定がない場合は，念の為，依頼者に対して再確認してほしい： 
> * モデルカードで推奨システムメッセージが明示されているにもかかわらず，指定されていない
> * モデルカードで推論モードをonにする条件が明示されているにもかかわらず，指定されていない

### 3.3 各モデル用のディレクトリについて
[2.5 評価結果の確認](#25-評価結果の確認)や[2.6 評価ログの確認](#26-評価ログの確認)で「各モデル用のディレクトリ」として \
`{STORE_PATH}/results/{provider}/{model_publisher}/{model_name}/{custom_settings}` というパスを記しているが，
このパスに含まれる各要素は以下の通りである．

| 表記 | 意味 |
| -- | -- |
| `{provider}` | モデルをserve（立てる）サービスの名前．vllmの場合は`hosted_vllm`，deepinfraの場合は`deepinfra`，openaiの場合は空文字，OpenRouter の場合は `openrouter` となる．|
| `{model_publisher}` | Huggingface Model ID を"/"で区切った時の前半部分．モデルを提供する団体を指す．（e.g. `tokyotech-llm`）
| `{model_name}` | Huggingface Model ID を"/"で区切った時の後半部分．モデル名を指す．（e.g. `Llama-3.1-Swallow-8B-Instruct-v0.3`）
| `{custom_settings}` | モデル固有の特別な生成条件．指定していない場合は空文字となる． |

なお，`{provider}` と `{custom_settings}` については空文字となりうるが，その場合パス内で空文字は端折られる． \
（e.g. `results//tokyotech-llm/swallow/` -> `results/tokyotech-llm/swallow`）

### 3.4 特定のproviderでエラーが出る場合の対応

特定のproviderでエラーが出る場合は，provider固有の制限に抵触している可能性がある．\
たとえば `deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8` では並列応答数 `n` に5以上を指定するとエラーを起こす（2025年6月時点）．   \
このような場合は，Jupyter Notebookなどを用いてproviderに適当なリクエストを投げて，調査を行なってほしい．

具体例は以下の通り．  
```
# deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 の例

import litellm

request_payload = {
    "model": "deepinfra/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    "messages": [
        {
            "role": "user",
            "content": (
                "こんにちは。なにかしゃべって"
            ),
        }
    ],
    "logprobs": None,
    "base_url": "https://api.deepinfra.com/v1/openai",
    "n": 2,
    "caching": False,
    "api_key": "DeepInfraのAPIキー",
    "max_completion_tokens": 512,
    "temperature": 1.0
}

responses = litellm.completion(**request_payload)
```

### 3.5 並列応答数 `n` を強制的に1にする方法

JHumanEval や LiveCodeBench のように "N回解いて正答率を測定する" ベンチマークでは，\
並列応答数 `n` をN（たとえばJHumanEvalならN=10）で設定している． \
providerが並列応答数を制限していてエラーになる場合などは generation_parameters に `max_n` の指定を追加してほしい．

具体例は以下の通り（上：generation_parametersに追記する例．下：手元で検証を行う例．）
```yaml
max_n_specified:
  max_n: "4"
  version: "1"
```

```
lighteval endpoint litellm \
    "model=$MODEL_NAME,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={temperature:0.2,top_p:0.95,max_n:4}" \
    "swallow|lcb:codegeneration_v5_6|0|0" \
    ...
```

### 3.6 評価がいつまでたっても終わらない場合

評価がいつまでたっても終わらない場合は，\
1 ) 並列処理数が少なすぎる 2 ) repetitionが多発してvLLMのスループットが極端に低下している または 3 ) 計算資源が足りない が疑われる． \
このような場合はGPUステータスやvLLMログを見て，スループットの極端な低下やKV Cache不足のWARNING多発など原因の手がかりを確認してほしい．
- Ref. [vLLM Optimization and Tuning](https://docs.vllm.ai/en/latest/configuration/optimization.htm)

#### 1) 並列処理数が少なすぎる

`nvidia-smi` コマンド等で表示できるGPUステータスで使用率が80%未満の場合は，推論リクエストの並列処理数を増やして高速化できる． \
．環境変数 `LITELLM_CONCURRENT_CALLS` で指定し，vLLMログに表示される "Avg generation throughput" の値が大きくなれば効果あり．  

>[!Note]
> デフォルト値は20．

並列処理数の目安は 8Bパラメータ を H100 MEM80GB x1 で推論するケースで 50--100 くらい．  
ぴったり最適化する意味はあまりないので，GPU枚数やGPUメモリ容量に比例および，モデルパラメータ数に反比例させて大雑把に調整するとよい．

OpenAIなどの外部プロバイダで推論する場合も同じく `LITELLM_CONCURRENT_CALLS` で並列処理数を増やせる．
APIのレートリミットを確認しながら50--100くらいに調整してみてほしい．

>[!Warning]
> 並列応答数 `n` > 1 のベンチマークについては，実際の並列処理数が `LITELLM_CONCURRENT_CALLS` / `n` と少なくなる．  
> 例えば n=10 なら `LITELLM_CONCURRENT_CALLS` で指定した並列処理数の10分の1になるので注意．

#### 2) repetitionが多発してスループットが極端に低下している

max-samples=10くらいに設定してモデルが生成した回答を眺めてみて，延々と同じ文字列を繰り返すrepetitionが起きているかを確認してほしい．  
経験的には推論型モデル，および LiveCodeBench や MATH-500 など数学・科学のベンチマークでrepetitionが起きやすい．  

>[!Note]
> repetitionの問題は，基本的には計算資源の増強で解決したい．\
> それが難しい場合には，依頼者に報告したうえで，`MAX_MODEL_LENGTH` を8,192まで下げることで解決を図ってほしい．\
> 推論型モデルの場合は temperature を0.6に上げたり top_p を0.95に下げたりする選択肢もあるが，モデル間の公平性が損なわれるので最後の手段である．

#### 3 ) 計算資源が足りない

vLLMログに `Aborted request` が出力されている，またはlightevalログに `Timeout` が出力されている場合は，推論に時間がかかりすぎてAPI呼び出しがタイムアウトしている．  
この場合は環境変数 `REQUEST_TIMEOUT` （単位は秒）に十分に大きな値を設定すること．

#### 4 ) [gpt-oss系列限定] コード生成タスクでBadRequestErrorが多発している

**このパターンは gpt-oss とその派生モデル かつ LiveCodeBench のようなコード生成タスクに限定される．** 

vLLMログに `BadRequestError` が大量に出力されている場合は，推論失敗によるリトライで時間を無駄にしている可能性がある．  
この場合は `max_n` を1にして，同時に `LITELLM_CONCURRENT_CALLS` を 100--200 にしてみてほしい．  

LiveCodeBenchや(J)HumanEvalは，通常は1リクエストで10応答 (n=10) を得るのだが，ひとつでも推論に失敗するとBadRequestErrorでのこり9個もぜんぶ破棄されるので
`n=1` にしてそのかわりに並列リクエスト数を増やすことで，破棄を減らすという対策である．  
推論失敗時に BadRequestError が起きるgpt-oss系列のみで有効な対策であり，gpt-oss系列以外には意味がない．  

### 3.7 vLLMのreasoning_parserが対応していない推論型モデル

vLLMには推論過程を除去する reasoning parser がビルトインされているが，parserが対応していない推論型モデルについてはRuntimeErrorが発生する．  

```
# parserが対応してないモデルでRuntimeErrorが発生した例．ここでは --reasoning-parser=deepseek_r1 を指定した．
RuntimeError: DeepSeek R1 reasoning parser could not locate think start/end tokens in the tokenizer!
```

推論過程の除去は必須なのでなんとかしてparserを適用しなくてはいけない．この場合は2つの方策がある．  

#### 3.7.1 モデル開発元が提供するカスタム reasoning parser を使用する

vLLM には，カスタム reasoning parser を渡す機能がある．具体的には `vllm serve --reasoning-parser-plugin` 引数に ReasoningParserクラスを実装した `.py` ファイルを渡す仕組みである．  
モデル開発元が reasoning_parser.py 等のファイルを提供している場合は，この機能を使えばよい．例：[nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese/blob/main/nemotron_nano_v2_reasoning_parser.py)

自動評価スクリプトで vLLM にカスタム reasoning parser を渡すには，提供された .py ファイルを `/scripts/generation_settings/reasoning_parser/`に配置するだけでよい．  
ただしファイル名は `{parser名}_reasoning_parser.py`とすること．例：`scripts/generation_settings/reasoning_parser/llmjp4_reasoning_parser.py`  
あとはふだん通り，custom_model_settings の reasoning parser 引数に parser名を指定できるようになる．  

配置済みのカスタム reasoning parserは以下の通り（2026年4月現在）．  

|reasoning_parser|適用可能なモデルの例|配布元|
|--|--|--|
|llmjp4|`llm-jp/llm-jp-4-*-thinking`|[llmjp4_reasoning_parser.py](https://github.com/llm-jp/llm-jp-4-cookbook)|
|nemotron_nano_v2|`nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese`|[nemotron_nano_v2_reasoning_parser.py](https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2-Japanese)|

#### 3.7.2 自分でカスタム reasoning parser を実装する

モデル提供側が助けてくれないときは自力でなんとかするしかないが，parserの実装は楽ではない．  
そこで lighteval内部で vLLM の返り値に対して カスタム reasoning parser を適用する仕組みを作ってあるので，文字列処理でなんとかなるケースはこれを活用してほしい．Ref. [reasoning_parserディレクトリ](https://github.com/swallow-llm/swallow-evaluation-instruct-private/tree/main/lighteval/src/lighteval/models/vllm/reasoning_parser)  
この仕組みを使う場合は `vllm serve` には `--reasoning-parser` を指定せず，`lighteval endpoint litellm` の MODEL_ARGS に `reasoning_parser` を指定して実行する．  

```
# 実行のイメージ
vllm serve
(--reasoning-parser を指定しない)

lighteval endpoint litellm \
"model=モデル名,reasoning_parser={parser名},generation_parameters=..." \
(以下略)
```

この仕組みで使える実装済みのparserは以下の通り（2026年4月現在）．  

|reasoning_parser|推論過程のマークアップ|モデルの例|
|--|--|--|
|deepseek_r1_markup|"\<think\>,\<\/think\>"|Llama-3.1-Nemotron, MiMo-7B-RL|

>[!Note]
> custom_model_settings では vLLM ビルトインの parsre と独自実装の parser を区別することなく指定することができる． \
> ただし，新しい parser を追加した際には [common_fungs/classify_reasoning_parser](https://github.com/swallow-llm/swallow-evaluation-instruct-private/blob/0d91b7301f66bf46cf55424bf69d16b76c495293/scripts/qsub/common_funcs.sh#L329) に適宜反映して欲しい．

### 3.8 Baseモデルを無理やり評価する

swallow suite に実装されているベンチマークはすべて zero-shot かつ "考えてから解く" Instructモデルを想定したスタイルになっていて，"続きを予測する" Baseモデルを想定したスタイルではない．\
しかしMid-trainingのように事後学習を意識した事前学習を研究・実践する場合は，Baseモデルに対しても"考えてから解く"スタイルで評価したいことがあるかもしれない． \
このようなユースケースを想定して，本節ではBaseモデルを無理やり評価する方法を説明する． \
ただし **停止条件を設定するのが難しく**，また，考えてから解くスタイルの適否はモデル次第なので，性質をよく理解しているモデルの研究用途でのみ使うことをおすすめする．  

Baseモデルを無理やり評価する方法は，Backendにより異なる．  

#### vllm serve & litellm backend の場合
role:user の content を平文化する chat template `./resources/chat_template_base_model.jinja` を vllm serve の chat-template 引数に渡す必要がある．  

```
# chat templateを持たないBaseモデル
MODEL="Qwen/Qwen2.5-1.5B"
HOST=localhost
PORT=9001
BASE_URL="http://$HOST:$PORT/v1"
API_KEY="dummy"

# vllmでホスティング
vllm serve $MODEL \
--chat-template ./resources/chat_template_base_model.jinja \
--host "$HOST" \
--port "$PORT"
```

Chat template の振る舞いを目視で確かめたい方は [Chat Template Playground](https://huggingface.co/spaces/huggingfacejs/chat-template-playground) を使ってほしい．  

lightevalの引数はいつもどおりの実行をすればよい．  
必要であれば `generation_parameters.stop_tokens` に，カンマ区切りで停止文字列を指定することもできる．Ref. [LiteLLM仕様書](https://docs.litellm.ai/docs/completion/input)

```
lighteval endpoint litellm \
    "model=hosted_vllm/$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={stop_tokens:\"def ,__main__\"}" \
    "swallow|swallow_jhumaneval|0|0" \
    --output-dir "$OUTPUT_DIR" \
    --use-chat-template

# stop は str型にしているのでエスケープした二重引用符をつけてください．
```

#### vLLMをlightevalから直接起動する vllm backend の場合
`--use-chat-template` 引数を外してほしい．  

```
MODEL="Qwen/Qwen2.5-1.5B"
MAX_MODEL_LENGTH=8192
TEMPERATURE=0.2
TOP_P=0.95

MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=$MAX_MODEL_LENGTH,generation_parameters={temperature:$TEMPERATURE,top_p:$TOP_P}"
OUTPUT_DIR=data/evals/

# --use-chat-template を外す
lighteval vllm $MODEL_ARGS \
"swallow|swallow_jhumaneval|0|0" \
--output-dir $OUTPUT_DIR \
--save-details
```

### 推論型モデルの reasoning effort を指定する

「どのくらい深く考えるか」を制御できるo3のような推論型モデルは `generation_parameters.reasoning_effort` に `low,middle,high` などの値を指定できます．  
ただし reasoning_effort は以下のユースケースのみ対応しています．  

1. litellm で外部ProviderのAPI (例： OpenAI API) を指定する場合
2. lighteval から vLLM を直接起動する `litellm vllm` で実行する場合

gpt-oss のようなモデルを `vllm serve` でセルフホストする場合は `reasoning_effort` は指定できません．  

使用例は以下のとおりです．  

```
# OpenAI API で o3 を評価

API_KEY="{OpenAI APIキー}"
BASE_URL="https://api.openai.com/v1/"
MODEL="openai/o3-2025-04-16"

lighteval endpoint litellm \
"model=$MODEL,api_key=$API_KEY,base_url=$BASE_URL,generation_parameters={reasoning_effort:\"high\"}" \
"swallow|gpqa:diamond|0|0" \
--output-dir "$OUTPUT_DIR"
```

### 3.9 qsub_multiを使って複数モデルを効率的に評価する
#### 使い方
例えば、Pass@K, Maj@K を複数モデルに対して評価する方法を例に説明します。
#### Step1. ```qsub_all.sh```の最下部で評価するタスクのみをコメントアウト (qsub_task ja gpqaなど)
<img width="322" height="718" alt="image" src="https://github.com/user-attachments/assets/ddb95d8b-f07d-44b4-b04d-cd3f65eb14d4" />

#### Step2. ```qsub_multi.sh```　の ```MODEL_NAMES``` に評価したいモデル名を追加
<img width="381" height="310" alt="image" src="https://github.com/user-attachments/assets/d57993a5-ea45-41cb-b28c-6af1c6b72931" />

#### Step3. ```scripts/generation_settings/generation_settings.json``` に評価したいモデルの設定がなければ追加
> 多くのモデルは登録済みなので、このStep3をスキップし、Step4を実行してみてエラーが出たら対応するくらいが良い。
> 追加する場合は他のモデルの設定を参考にする。node_kindは["rt_HG", "rt_HF", "rt_HC"]のいずれかを指定。
> custom_settingsは必要に応じて設定名を入れる。
> node_kindはtsubameで使う場合も[rt_HG, rt_HF, rt_HC]で指定し、qsub_multi内で自動で[node_q, node_f, cpu_16]に変換される。

<img width="445" height="289" alt="image" src="https://github.com/user-attachments/assets/3c05fb53-a428-49f7-a300-7a89e1ab212d" />

#### Step4. ```bash scripts/qsub/qsub_multi.sh```
ターミナルで以下を実行すればOK
```bash
bash scripts/qsub/qsub_multi.sh
```
