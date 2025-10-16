from argparse import ArgumentParser
from pathlib import Path
import json
import os
import re

out_dir = Path(__file__).resolve().parent / "issues_status"


def issue_create(issue_id, model_names, tasks, custom_settings=""):
    out_file = out_dir / f"{issue_id}.json"
    try:
        out_file.touch(exist_ok=True)
    except:
        print(f"‼️ Issue {issue_id} already exists.")
        exit(1)
    data = {"models": {model_name: {task: {} for task in tasks} for model_name in model_names}}
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4) 

def job_submitted(issue_id, model_id, task_id, job_id, custom_settings=""):
    out_file = out_dir / f"{issue_id}.json"
    with open(out_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    task_jobs = data.get("models", {}).get(model_id, {}).get(task_id, None)
    if task_jobs == None:
        task_jobs[job_id] = {"version": 1}
    else:
        task_jobs[job_id] = {"version": len(task_jobs) + 1}
    task_jobs[job_id]["status"] = "qw"
    data["custom_settings"] = custom_settings

    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def register_status(issue_id, qstat_log):
    qstat_log_split = qstat_log.split()
    out_file = out_dir / f"{issue_id}.json"
    result_dir = Path(__file__).resolve().parents[4] / "results"

    with open(out_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    models = data["models"]
    custom_settings = data["custom_settings"]
    for model_name, model_tasks in models.items():
        for task_name, jobs in model_tasks.items():
            for job_id, job in jobs.items():
                if job_id in qstat_log_split:
                    job["status"] = qstat_log_split[qstat_log_split.index(job_id) + 4]
                else:
                    lang, task = task_name.split()
                    task_dir = "mt_bench" if task == "mtbench" else task
                    model_name_relative = model_name[1:] if model_name.startswith('/') else model_name
                    # For TSUBAME
                    result_o_file = result_dir / "hosted_vllm" / model_name_relative / custom_settings / lang / task_dir / f"{lang}_{task}.o{job_id}"
                    if not result_o_file.is_file():
                        # For ABCI
                        result_o_file = result_dir / "hosted_vllm" / model_name_relative / custom_settings / lang / task_dir / f"{job_id}.OU"
                    if not result_o_file.is_file():
                        # deleted
                        job["status"] = "deleted"
                        continue

                    # Open .o file
                    with open(result_o_file, "r", errors="ignore") as f:
                        o_content = f.read()

                    is_finished = re.search(r"✅ Result aggregation was successfully done.", o_content)
                    error_occurred = re.search(r"Traceback \(most recent call last\):", o_content)
                    if error_occurred:
                        job["status"] = "error"
                    else:
                        # For TSUBAME
                        result_e_file = result_dir / "hosted_vllm" / model_name_relative / custom_settings / lang / task_dir / f"{lang}_{task}.e{job_id}"
                        if not result_e_file.is_file():
                            # For ABCI
                            result_e_file = result_dir / "hosted_vllm" / model_name_relative / custom_settings / lang / task_dir / f"{job_id}.ER"
                        with open(result_e_file, "r", errors="ignore") as f:
                            e_content = f.read()
                        error_occurred = re.search(r"Traceback \(most recent call last\)", e_content)
                        if error_occurred:
                            job["status"] = "error"
                        elif is_finished:
                            job["status"] = "done"
                        else: 
                            job["status"] = "timeout"

    # register status to json file
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        

def create_resub(issue_id):
    script_dir = Path(__file__).resolve().parent
    json_file = script_dir / "issues_status" / f"{issue_id}.json"
    out_dir = script_dir / "resub"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{issue_id}.csv"

    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)
    custom_settings = data["custom_settings"]
    models = data["models"]

    with open(out_file, "w") as f:
        for model_name, model_tasks in models.items():
            for task_name, jobs in model_tasks.items():
                if len(jobs) == 0:
                    f.write(f"{model_name}, {task_name}\n")
                    continue
                status_list = [job["status"] for job_id, job in jobs.items()]
                if not "done" in status_list:
                    version_list = [job["version"] for job_id, job in jobs.items()]
                    ver_max = max(version_list)
                    ver_max_index = version_list.index(ver_max)
                    job = tuple(jobs.values())[ver_max_index]
                    job_id = tuple(jobs.keys())[ver_max_index]
                    lang, task = task_name.split()
                    
                    if not job["status"] in ["r", "qw", "R", "Q"]:
                        f.write(f"{model_name}, {task_name}\n")
    print(f"👀 Created resub tasks:\n\t{out_file}")
                

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--action", type=str, required=True)
    parser.add_argument("--issue_id", type=str, required=True)
    parser.add_argument("--model_id", type=str, default="")
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument("--job_id", type=str, default="")
    parser.add_argument("--status", type=str, default="")
    parser.add_argument("--qstat_log", type=str, default="")
    parser.add_argument("--custom_settings", type=str, default="")
    args = parser.parse_args()


    if args.action == "register_status":
        register_status(args.issue_id, args.qstat_log)
        create_resub(args.issue_id)
    elif args.action == "create_resub":
        create_resub(args.issue_id)
    else:
        # Get model names
        MODEL_NAMES_JSON = os.environ.get("MODEL_NAMES_JSON")
        model_names = json.loads(MODEL_NAMES_JSON)
        # Get tasks
        TASKS_JSON = os.environ.get("TASKS_JSON")
        tasks = json.loads(TASKS_JSON)
        if args.action == "issue_create":
            assert model_names != [] and tasks != [], "💀 Make sure that model_names and tasks are set."
            issue_create(args.issue_id, model_names, tasks)
        elif args.action == "job_submitted":
            assert args.model_id != "" and args.task_id != "" and args.job_id != ""
            job_submitted(args.issue_id, args.model_id, args.task_id, args.job_id, args.custom_settings)
        
