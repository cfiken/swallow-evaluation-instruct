from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.extended.ifbench.main import ifbench_metrics, ifbench_prompt

# IFBench
ifbench = LightevalTaskConfig(
    name="ifbench_singleturn",
    prompt_function=ifbench_prompt,
    suite=["swallow"],
    hf_repo="allenai/IFBench_test",
    hf_subset="default",
    metric=[ifbench_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)
