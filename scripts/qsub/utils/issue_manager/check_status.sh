#!/bin/bash
# Usage: bash check_status.sh {issue_id}

set -euo pipefail

ISSUE_ID=$1

# Load .env 
source "$(dirname "$0")/../../../../.env"

# QSTAT_LOG=$(qstat)
QSTAT_LOG=$( cat /gs/fs/tga-okazaki/matsushita/swallow-evaluation-instruct-private/scripts/qsub/utils/issue_manager/test.txt)

# Check and register status
python "${REPO_PATH}/scripts/qsub/utils/issue_manager/manage_jobs.py" --action register_status --issue_id "${ISSUE_ID}" --qstat_log "${QSTAT_LOG}"

python "${REPO_PATH}/scripts/qsub/utils/issue_manager/visualize_status.py" --issue_id "${ISSUE_ID}"

