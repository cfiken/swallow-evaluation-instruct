from lighteval.tasks.lighteval_task import LightevalTaskConfig
import lighteval.tasks.default_prompts as prompt
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.passk_utils import create_passk_metrics, powers_of_two_up_to_n
from copy import deepcopy

math_500_swallow = LightevalTaskConfig(
    name="math_500",
    suite=["swallow"],
    prompt_function=prompt.math_500,
    hf_repo="HuggingFaceH4/MATH-500",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split=None,
    few_shots_select=None,
    generation_size=None,  # swallow用に変更
    metric=[Metrics.latex_gold_metric],
    version=1,
)

# Pass@K variant
lst_math_500_swallow_passk = []
dict_passk_metric = {}
for num_samples in [16, 32, 64, 128, 256]:
    lst_k = powers_of_two_up_to_n(num_samples)
    # Metricsクラスに属するSampleLevelMetricを指定する場合は .value をつける
    dict_passk_metric[num_samples] = create_passk_metrics(base_metric=Metrics.latex_gold_metric.value, k_values=lst_k, num_samples=num_samples)
    
    task_config = deepcopy(math_500_swallow)
    task_config.name = f"math_500_{num_samples}"
    task_config.metric = dict_passk_metric[num_samples]
    lst_math_500_swallow_passk.append(task_config)