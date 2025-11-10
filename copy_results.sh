#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-}"
if [[ -z "$SRC_ROOT" ]]; then
  echo "Usage: $0 /path/to/.../reasoning_swallow"
  exit 1
fi

DST_ROOT="/groups/gag51395/share/se_eval_details"

mkdir -p "$DST_ROOT"

tasks=(
  "japanese_mt_bench"
  "mifeval_ja"
  "jamcqa"
  "wmt20:en-ja"
  "wmt20:ja-en"
  "mmlu_prox_japanese"
  "swallow_gpqa_ja"
  "math_100_japanese"
  "swallow_jhumaneval"
  "english_mt_bench"
  "hellaswag"
  "mmlu_pro_english"
  "gpqa:diamond"
  "math_500"
  "aime"
  "lcb:codegeneration_v5_v6"
)

# reasoning_swallow 直下の日付ディレクトリ一覧
mapfile -t all_date_dirs < <(
  find "$SRC_ROOT" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' | sort
)

if [[ "${#all_date_dirs[@]}" -eq 0 ]]; then
  echo "No subdirectory found under $SRC_ROOT" >&2
  exit 1
fi

echo "Found date dirs:"
printf '  %s\n' "${all_date_dirs[@]}"
echo

not_found_tasks=()

for task in "${tasks[@]}"; do
  latest_dir_for_task=""

  # 各 task について、その task 名（＋サブタスク）を含むファイルがある「最新」ディレクトリを探す
  for d in "${all_date_dirs[@]}"; do
    mapfile -t found < <(
      find "$SRC_ROOT/$d" -maxdepth 1 -type f -name "*|${task}*|*.parquet" -print
    )
    if ((${#found[@]} > 0)); then
      latest_dir_for_task="$d"
    fi
  done

  if [[ -z "$latest_dir_for_task" ]]; then
    echo "[SKIP] No files found for task: ${task}"
    not_found_tasks+=("$task")
    continue
  fi

  echo "[TASK] ${task} -> latest dir: ${latest_dir_for_task}"
  SRC_DIR="$SRC_ROOT/$latest_dir_for_task"

  # 該当タスク + サブタスクのファイルを全部コピー
  while IFS= read -r file; do
    [[ -z "$file" ]] && continue

    # details/ より後ろを取り出して archive 配下にぶら下げる
    # 例:
    #  file = /.../lighteval/outputs/details/hosted_vllm/deepseek-ai/.../reasoning_swallow/2025-.../details_swallow|...parquet
    #  rel  = hosted_vllm/deepseek-ai/.../reasoning_swallow/2025-.../details_swallow|...parquet
    rel_path="${file#*details/}"

    dest="$DST_ROOT/$rel_path"

    mkdir -p "$(dirname "$dest")"
    cp -p "$file" "$dest"
    echo "  Copied: $file"
    echo "       -> $dest"
  done < <(find "$SRC_DIR" -maxdepth 1 -type f -name "*|${task}*|*.parquet" -print)
done

echo
if [[ "${#not_found_tasks[@]}" -gt 0 ]]; then
  echo "❌ Not found tasks:"
  for t in "${not_found_tasks[@]}"; do
    echo "  ❌ $t"
  done
else
  echo "✅ All tasks found and copied successfully!"
fi
