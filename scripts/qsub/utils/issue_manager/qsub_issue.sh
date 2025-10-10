#!/bin/bash
# Usage: bash qsub_issue.sh {issue_number}

set -euo pipefail

source "$(dirname $0)/issues/$1"

# Load .env and define dirs
source "$(dirname "$0")/../../../../.env"
case $CUSTOM_SETTINGS in
    "") CUSTOM_SETTINGS_SUBDIR="" ;;
    *) CUSTOM_SETTINGS_SUBDIR="/${CUSTOM_SETTINGS}" ;;
esac
case $PROVIDER in
    openai) PROVIDER_SUBDIR="" ;;
    vllm) PROVIDER_SUBDIR="hosted_vllm/" ;;
    deepinfra) PROVIDER_SUBDIR="deepinfra/" ;;
    *) echo "❌ unknown provider ${PROVIDER}"; exit 1 ;;
esac
# Load task-definition and common functions
source "$(dirname "$0")/../../conf/load_config.sh"
source "${REPO_PATH}/scripts/qsub/common_funcs.sh"

# Optional Args (whose values can be empty)
## When a value is empty, do not pass its arg name either to avoid an arg parsing error.
OPTIONAL_ARGS=()
if [[ -n "${CUSTOM_SETTINGS}" ]]; then
  OPTIONAL_ARGS+=(--custom-settings ${CUSTOM_SETTINGS})
fi
if [[ -n "${MAX_SAMPLES}" ]]; then
  OPTIONAL_ARGS+=(--max-samples ${MAX_SAMPLES})
fi

# Check service
check_service "${SERVICE}"

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    RESULTS_DIR="${REPO_PATH}/results/${PROVIDER_SUBDIR}${MODEL_NAME}${CUSTOM_SETTINGS_SUBDIR}"
    SCRIPTS_DIR="${REPO_PATH}/scripts/qsub"

    # Set common qsub args
    common_qsub_args=(
    --node-kind ${NODE_KIND}
    --provider ${PROVIDER}
    --model-name ${MODEL_NAME}
    --repo-path ${REPO_PATH}
    --service ${SERVICE}
    )

    # Pre-download the model
    if [ ${PREDOWNLOAD_MODEL} = "true" ]; then
    if [ ${PROVIDER} != "vllm" ]; then
        echo "☠️ Error: Pre-downloading is only supported for vLLM. Please set PREDOWNLOAD_MODEL=false."
        exit 1
    fi
    source "${REPO_PATH}/.common_envs/bin/activate"
    echo "🤖 Downloading ${MODEL_NAME} ..."
    huggingface-cli download $MODEL_NAME --cache-dir $HUGGINGFACE_CACHE --token $HF_TOKEN
    deactivate
    echo "✅ \`${MODEL_NAME}\` was successfully downloaded at \`${HUGGINGFACE_CACHE}\`."
    else
    echo "⏭️ Skipping pre-downloading model."
    fi

    # Define qsub-function
    last_submit_time=""
    qsub_task() {
    # Get args
    local lang=$1 task=$2

    # Get task-specific args
    result_dir=$(task_result "${lang}_${task}")
    task_name=$(task_script "${lang}_${task}")
    task_framework=$(task_framework "${lang}_${task}")
    [[ -z $task_name || -z $result_dir || -z $task_framework ]] && { echo "❌ Unknown task ${lang}_${task}"; exit 1; }

    # Set an outdir 
    OUTDIR="${RESULTS_DIR}/${result_dir}"
    mkdir -p "${OUTDIR}"

    # Safety check for local jobs
    if [[ "${SERVICE}" == "local" && -n "${last_submit_time}" ]]; then
        now=$(date +%s); elapsed=$(( now - last_submit_time ))
        if (( $elapsed < 30 )); then
        echo "💀 Error: Local jobs cannot be submitted continuously. Please reset CUDA_VISIBLE_DEVICES appropriately and submit one task at a time."
        exit 1
        fi
    fi

    # Submit a job
    local job_name="${lang}_${task}"
    case $SERVICE in
        "tsubame")
        h_rt=$(hrt "${NODE_KIND}" "${lang}_${task}") || { echo "❌ Cound not get h_rt for ${lang}_${task} on ${NODE_KIND}"; exit 1; }
        qsub -g "${TSUBAME_GROUP}" -l "${NODE_KIND}"=1 -p "${PRIORITY}" -N "${job_name}" -l h_rt="${h_rt}" -o "${OUTDIR}" -e "${OUTDIR}" "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
            --task-name "${task_name}" "${common_qsub_args[@]}" "${OPTIONAL_ARGS[@]}"
        ;;

        "abci")
        wlt=$(walltime "${NODE_KIND}" "${lang}_${task}") || { echo "❌ Cound not get walltime for ${lang}_${task} on ${NODE_KIND}"; exit 1; }
        qsub -P "${ABCI_GROUP}" -q "${NODE_KIND}" -l select=1 -N "${job_name}" -l walltime="${wlt}" -o "${OUTDIR}" -e "${OUTDIR}" -- "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
            --task-name "${task_name}" "${common_qsub_args[@]}" --stdout-stderr-dir "${OUTDIR}" "${OPTIONAL_ARGS[@]}"
        ;;

        "local")
        set_random_job_id
        local session_name="${job_name}_${JOB_ID}"
        tmux new-session -d -s "${session_name}" \
            env ${CUDA_VISIBLE_DEVICES:+CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}} \
            bash "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
            --task-name "${task_name}" "${common_qsub_args[@]}" --stdout-stderr-dir "${OUTDIR}" --custom-job-id "${JOB_ID}" "${OPTIONAL_ARGS[@]}"
        echo "✅ Local job ${job_name} was successfully submitted to tmux session ${session_name}."
        ;;
    esac
    last_submit_time=$(date +%s)
    }

    ########################################################

    # Submit tasks
    echo "🚀 Submitting tasks..."

    ## Japanese
    # qsub_task ja gpqa
    # qsub_task ja jemhopqa_cot
    # qsub_task ja jamcqa
    # qsub_task ja math_100
    # qsub_task ja mmlu_prox
    # qsub_task ja mtbench
    # qsub_task ja wmt20_en_ja
    # qsub_task ja wmt20_ja_en
    # qsub_task ja humaneval
    # qsub_task ja mifeval

    # ## English
    # qsub_task en hellaswag
    # qsub_task en mtbench
    # qsub_task en gpqa_diamond
    # qsub_task en math_500
    # qsub_task en aime_2024_2025
    # qsub_task en livecodebench_v5_v6
    # qsub_task en mmlu_pro

    ## Optional
    # qsub_task ja mmlu
    # qsub_task ja jemhopqa
    # qsub_task ja jamcqa_cot
    # qsub_task en mmlu
    # qsub_task en mmlu_prox
    # qsub_task en humaneval
    # qsub_task en humanevalplus
    # qsub_task ja jgpqa_diamond
    # qsub_task ja jgpqa_diamond_n16
    # qsub_task ja gpqa_n16
    # qsub_task ja math_100_n16
    # qsub_task en gpqa_diamond_n16
    # qsub_task en math_500_n16
    # qsub_task en aime_2024_2025_n16
done

# readarray -t JOB_IDS < <(python3 /home/ach17941yz/swallow-evaluation-instruct-private/scripts/qsub/utils/issue_manager/check_status.py)
# for job_id in "${JOB_IDS[@]}"; do
#     echo "$job_id"
# done