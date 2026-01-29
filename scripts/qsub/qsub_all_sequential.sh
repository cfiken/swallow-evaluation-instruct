#!/usr/bin/env bash
# How-to-use:
# 1. Modify corresponding variables in qsub_all.sh to $1, $2, ... respectively to receive arguments from this script.
# 2. Set other parameters in qsub_all.sh as needed.
# 3. Copy and paste the target models given in a request for evaluation into the target_models array.
# 4. Run this code by `bash scripts/qsub/qsub_all_sequential.sh`.
set -euo pipefail

target_models=(
dummy_model/delete_us_before_evaluation
dummy_model/all_you_have_to_do_is_copy_and_paste

dummy_model/empty_lines_are_ignored_so_don_t_worry

dummy_model/we_hope_this_code_is_useful_for_you
dummy_model/have_a_nice_day
)

node_kind="node_q"

custom_setting=""

for model in ${target_models[@]}; do
    qsub_args=("${node_kind}" "${model}" "${custom_setting}")
    bash scripts/qsub/qsub_all.sh "${qsub_args[@]}"
done
