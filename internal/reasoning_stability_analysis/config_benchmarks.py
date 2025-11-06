
from aggregation_functions import extractive_match_metric

BENCHMARKS = {
    "swallow|jamcqa|0": extractive_match_metric,
    "swallow|mmlu_prox_japanese|0": extractive_match_metric,
    "swallow|swallow_gpqa_ja|0": extractive_match_metric,
    "swallow|math_100_japanese|0": extractive_match_metric,
}