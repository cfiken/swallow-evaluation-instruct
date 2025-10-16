from argparse import ArgumentParser
from pathlib import Path
import json
import os


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--issue_id", type=str, required=True)
    args = parser.parse_args()
    
    issue_id = args.issue_id

    script_dir = Path(__file__).resolve().parent
    json_file = script_dir / "issues_status" / f"{issue_id}.json"
    out_file = script_dir / "visualized" / f"{issue_id}.txt"
    result_dir = Path(__file__).resolve().parents[4] / "results"
    (script_dir / "visualized").mkdir(exist_ok=True)

    with open(json_file, "r", encoding='utf-8') as f:
        data = json.load(f)

    custom_settings = data["custom_settings"]
    models = data["models"]

    with open(out_file, "w") as f:
        f.write(f"Issue_id: {issue_id}\nCustom_setting: {custom_settings}\n")
        f.write(f"================================================================================\n")
        for i, (model_name, model_tasks) in enumerate(models.items()):
            if i > 0:
                f.write(f"--------------------------------------------------------------------------------\n")
            f.write(f"Model: {model_name}\n")
            for task_name, jobs in model_tasks.items():
                if len(jobs) == 0:
                    f.write(f"\t⬜️ not submitted\ttask: [{task_name}]")
                status_list = [job["status"] for job_id, job in jobs.items()]
                if "done" in status_list:
                    job_id = tuple(jobs.keys())[status_list.index("done")]
                    f.write(f"\t🟢 done   \tjob_id: [{job_id}]\ttask: [{task_name}]\n")
                else:
                    version_list = [job["version"] for job_id, job in jobs.items()]
                    ver_max = max(version_list)
                    ver_max_index = version_list.index(ver_max)
                    job = tuple(jobs.values())[ver_max_index]
                    job_id = tuple(jobs.keys())[ver_max_index]
                    lang, task = task_name.split()
                    
                    if job["status"] == "deleted":
                        f.write(f"\t🔳 deleted\tjob_id: [{job_id}]\ttask: [{task_name}]\n")
                    elif job["status"] in ["qw", "Q"]:
                        f.write(f"\t🟤 queue  \tjob_id: [{job_id}]\ttask: [{task_name}]\n")
                    else:
                        task_dir = "mt_bench" if task == "mtbench" else task
                        model_name_relative = model_name[1:] if model_name.startswith('/') else model_name
                        # For TSUBAME
                        result_o_file = result_dir / "hosted_vllm" / model_name_relative / custom_settings / lang / task_dir / f"{lang}_{task}.o{job_id}"
                        if not result_o_file.is_file():
                            # For ABCI
                            result_o_file = result_dir / "hosted_vllm" / model_name_relative / custom_settings / lang / task_dir / f"{job_id}.OU"
                        if job["status"] in ["r", "R"]:
                            f.write(f"\t🔵 running")
                        elif job["status"] == "timeout":
                            f.write(f"\t🟨 timeout")
                        elif job["status"] == "error":
                            f.write(f"\t🟥 error  ")
                        # f.write(f"\tjob_id: [{job_id}]\ttask: [{task_name}]\tfile://{result_o_file}\n")
                        f.write(f"\tjob_id: [{job_id}]\ttask: [{task_name}]\n")

        f.write(f"================================================================================\n")
    print(f"💡Visualized issue status:\n\t{out_file}")
