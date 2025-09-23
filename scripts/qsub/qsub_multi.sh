#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# このスクリプトは qsub_all.shを複数モデルに対して実行するためのものです。
# 現在はabci,tsubameのみ対応しています。(localは非対応)

# 使い方
# 1. qsub_all.shの最下部で評価するタスクのみをコメントアウト (qsub_task ja gpqaなど)
# 2. 下の MODEL_NAMES に評価したいモデル名を追加
# 3. scripts/generation_settings/generation_settings.json に評価したいモデルを追加 ( 追加する場合は他のモデルの設定を参考に。node_kindは["rt_HG", "rt_HF", "rt_HC"]のいずれかを指定, custom_settingsは必要に応じて設定名を入れる。node_kindはtsubameで使う場合も[rt_HG, rt_HF, rt_HC]で指定し、qsub_multi内で自動で[node_q, node_f, cpu_16]に変換されます。)
# 4. bash scripts/qsub/qsub_multi.sh

######## Common Settings (固定としてここで定義) ########

# 評価したいモデル名を改行区切りで追加
MODEL_NAMES=(
  "tokyotech-llm/example-8B-Instruct"
  "tokyotech-llm/example-12B-Instruct"
) 

######## Common Settings ########
PROVIDER="vllm"                 # ["vllm"] 
SERVICE="abci"                  # ["abci","tsubame"]
PREDOWNLOAD_MODEL="true"        # ["true","false"] falseだと大量のモデルを同時にダウンロードすることになり、HFのレートリミットに引っかかる可能性がある
MAX_SAMPLES=""
PRIORITY="-5"
CUDA_VISIBLE_DEVICES=""
######################################################
# これより下は基本的に変更不要

JOBS_FILE="../generation_settings/generation_settings.json"
SCRIPT="./qsub_all.sh"
PASS_ARG="enable_env" 
ALLOW_KEYS=("$@")
if [[ "${#ALLOW_KEYS[@]}" -eq 0 ]]; then
  ALLOW_KEYS=("${MODEL_NAMES[@]}")
fi

# jq チェック
if ! command -v jq >/dev/null 2>&1; then
  echo "❌ jq is required but not installed." >&2
  exit 1
fi

# JSONが空/不正なら落とす（オブジェクト前提）
if ! jq -e 'type == "object"' "$JOBS_FILE" >/dev/null; then
  echo "❌ $JOBS_FILE must be a JSON object keyed by model_name." >&2
  exit 1
fi

job_idx=0
for key in "${ALLOW_KEYS[@]}"; do
  # モデル名がJSONに存在しない場合はストップ
  model="$key"
  if ! jq -e --arg k "$key" 'has($k)' "$JOBS_FILE" >/dev/null; then
    echo "⚠️  ERROR: key \"$key\" not found in $JOBS_FILE" >&2

    # 「tokyotech-llm/xxxx-iterNNN」形式なら、-iter以降を落として再チェック
    if [[ "$key" =~ ^tokyotech-llm/.+-iter[0-9]+$ ]]; then
      alt_key="${key%-iter*}"

      if jq -e --arg k "$alt_key" 'has($k)' "$JOBS_FILE" >/dev/null; then
        echo "   ⚠️  Note: using alternative key \"$alt_key\" instead." >&2
        key="$alt_key"   # 以降このキーで参照
      else
        echo "❌ Also not found with alternative key \"$alt_key\". Exiting." >&2
        exit 1
      fi

    else
      echo "❌ No matching fallback rule for \"$key\". Exiting." >&2
      exit 1
    fi
  fi

  echo "======== 🧩 Job #$job_idx — $model ========"
  NK="$(jq -er --arg k "$key" '.[$k].node_kind' "$JOBS_FILE")" || {
    echo "❌ Error: key \"$key\" has no 'node_kind' in $JOBS_FILE" >&2
    exit 1
  }


  # providerがtsubameの場合はnode_kindの変換をする rt_HG -> node_q, rt_HF -> node_f
  if [[ "$SERVICE" == "tsubame" ]]; then
    case $NK in
      rt_HG) NK="node_q" ;;
      rt_HF) NK="node_f" ;;
      rt_HC) NK="cpu_16" ;;
      *) echo "❌ unknown node_kind ${NK} for tsubame"; exit 1 ;;
    esac
  fi

  CUSTOM_SETTINGS="$(jq -r --arg k "$key" '.[$k].custom_settings // empty' "$JOBS_FILE")"

  # 環境変数として定義して、qsub_all.shではこの設定で上書きして実行する
  envs=(
    "ENV_MODEL_NAME=$model"     
    "ENV_NODE_KIND=$NK"
    "ENV_CUSTOM_SETTINGS=$CUSTOM_SETTINGS"
    "ENV_PROVIDER=$PROVIDER"
    "ENV_PREDOWNLOAD_MODEL=$PREDOWNLOAD_MODEL"
    "ENV_MAX_SAMPLES=$MAX_SAMPLES"
    "ENV_SERVICE=$SERVICE"
    "ENV_PRIORITY=$PRIORITY"
    "ENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  )

  # デバッグ表示
  printf 'env '
  printf "%q " "${envs[@]}"
  printf 'bash %q %q\n' "$SCRIPT" "$PASS_ARG"

  # 実行
  env "${envs[@]}" bash "$SCRIPT" "$PASS_ARG"

  job_idx=$((job_idx + 1))

done