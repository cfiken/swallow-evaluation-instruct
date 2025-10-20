#!/bin/bash
# Usage: bash qsub_issue.sh {issue_id}

set -euo pipefail

ISSUE_ID=$1

source "$(dirname $0)/issues/${ISSUE_ID}"

# Load .env and define dirs
source "$(dirname "$0")/../../../../.env"
mkdir -p "${REPO_PATH}/scripts/qsub/utils/issue_manager/issues_status"
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


# Register tasks
MODEL_NAMES_JSON=$(jq -c -n '$ARGS.positional' --args "${MODEL_NAMES[@]}")
TASKS_JSON=$(jq -c -n '$ARGS.positional' --args "${TASKS[@]}")

if [[ -f "${REPO_PATH}/scripts/qsub/utils/issue_manager/issues_status/${ISSUE_ID}.json" ]]; then
    NEW_ISSUE=0
else
    python "${REPO_PATH}/scripts/qsub/utils/issue_manager/manage_jobs.py" --action issue_create --issue_id ${ISSUE_ID}
    NEW_ISSUE=1
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
        QSUB_MESSAGE=$(qsub -g "${TSUBAME_GROUP}" -l "${NODE_KIND}"=1 -p "${PRIORITY}" -N "${job_name}" -l h_rt="${h_rt}" -o "${OUTDIR}" -e "${OUTDIR}" "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
        --task-name "${task_name}" "${common_qsub_args[@]}" "${OPTIONAL_ARGS[@]}")
        
        echo ${QSUB_MESSAGE}
        QSUB_MESSAGE_ARRAY=($QSUB_MESSAGE)
        JOB_ID=${QSUB_MESSAGE_ARRAY[2]}
        if [ -z "$CUSTOM_SETTINGS" ]; then
            python "${REPO_PATH}/scripts/qsub/utils/issue_manager/manage_jobs.py" --action job_submitted --issue_id ${ISSUE_ID} --model_id ${MODEL_NAME} --task_id "${lang} ${task}" --job_id ${JOB_ID}
        else
            python "${REPO_PATH}/scripts/qsub/utils/issue_manager/manage_jobs.py" --action job_submitted --issue_id ${ISSUE_ID} --model_id ${MODEL_NAME} --task_id "${lang} ${task}" --job_id ${JOB_ID} --custom_settings ${CUSTOM_SETTINGS}
        fi
        ;;

        "abci")
        wlt=$(walltime "${NODE_KIND}" "${lang}_${task}") || { echo "❌ Cound not get walltime for ${lang}_${task} on ${NODE_KIND}"; exit 1; }
        QSUB_MESSAGE=$(qsub -P "${ABCI_GROUP}" -q "${NODE_KIND}" -l select=1 -N "${job_name}" -l walltime="${wlt}" -o "${OUTDIR}" -e "${OUTDIR}" -- "${SCRIPTS_DIR}/evaluate_${task_framework}.sh" \
            --task-name "${task_name}" "${common_qsub_args[@]}" --stdout-stderr-dir "${OUTDIR}" "${OPTIONAL_ARGS[@]}")
        echo ${QSUB_MESSAGE}
        JOB_ID=${QSUB_MESSAGE}
        if [ -z "$CUSTOM_SETTINGS" ]; then
            python "${REPO_PATH}/scripts/qsub/utils/issue_manager/manage_jobs.py" --action job_submitted --issue_id ${ISSUE_ID} --model_id ${MODEL_NAME} --task_id "${lang} ${task}" --job_id ${JOB_ID}
        else
            python "${REPO_PATH}/scripts/qsub/utils/issue_manager/manage_jobs.py" --action job_submitted --issue_id ${ISSUE_ID} --model_id ${MODEL_NAME} --task_id "${lang} ${task}" --job_id ${JOB_ID} --custom_settings ${CUSTOM_SETTINGS}
        fi
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

    ########################################################

    # Submit tasks
    echo "🚀 Submitting tasks..."

    for TASK in "${TASKS[@]}"; do
        if [[ "${NEW_ISSUE}" == "1" ]]; then
            # First submit
            qsub_task ${TASK}
        else
            CSV_FILE="${REPO_PATH}/scripts/qsub/utils/issue_manager/resub/${ISSUE_ID}.csv"
            if [[ -f "$CSV_FILE" ]]; then
                echo "♻️ Resubmitting tasks listed in ${CSV_FILE} ..."
            else
                bash "${REPO_PATH}/scripts/qsub/utils/issue_manager/check_status.sh" ${ISSUE_ID}
            fi
            if [[ -n $(grep "^$MODEL_NAME, $TASK" "$CSV_FILE") ]]; then
                # Re submit
                qsub_task ${TASK}
            fi
        fi
    done

done
echo "🎉 All tasks were successfully submitted."
