from copy import deepcopy

from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    multilingual_extractive_match_metric,
)
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language
from lighteval.metrics.sample_metric_utils import create_majk_metrics, create_passk_metrics, powers_of_two_up_to_n


POLYMATH_JAPANESE_QUERY_TEMPLATE = """
以下の数学の問題を解いてください。
出力の最後の行は、次の形式にしてください。

回答: $\\boxed{{ANSWER}}$

`ANSWER` には、解答となる数式または数値が入ります。

問題: 
{Question}
""".strip()

# 具体例
"""
以下の数学の問題を解いてください。
出力の最後の行は、次の形式にしてください。

回答: $\boxed{{ANSWER}}$

`ANSWER` には、問題の答えに対する最終的な数式または数値が入ります。

問題: 
ジャネットのアヒルは1日に16個の卵を生みます。ジャネットは毎朝朝食の一環で3個を消費し、毎日4個使って友達向けにマフィンを焼きます。残りを市場で1個あたり2ドルの価格で売ります。彼女は毎日市場でいくら手に入れていますか？
"""

def wrap_answer_with_latex_boxes(str_answer: str):
    TEMPLATE = "$\\boxed{{{ANSWER}}}$"

    if not str_answer.startswith("$\\boxed"):
        return TEMPLATE.format(ANSWER=str_answer)
    else:
        return str_answer


def polymath_japanese_prompt_fn(line, task_name: str = None):
    difficulty = None
    if isinstance(line.get("id"), str) and "-" in line["id"]:
        difficulty = line["id"].split("-", 1)[0]

    return Doc(
        task_name=task_name,
        query=POLYMATH_JAPANESE_QUERY_TEMPLATE.format(Question=line["question"]),
        choices=[wrap_answer_with_latex_boxes(line["answer"])],
        gold_index=0,
        specific={"difficulty": difficulty},
    )


# Evaluation metric
# 回答スパン抽出：数式 (LatexExtractionConfig) と 数量表現 (ExprExtractionConfig) を併用
latex_gold_metric = multilingual_extractive_match_metric(
    language=Language.ENGLISH,
    fallback_mode="first_match",
    precision=5,
    gold_extraction_target=(LatexExtractionConfig(),),
    # Match boxed first before trying other regexes
    pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)),
    aggregation_function=max,
)

POLYMATH_JAPANESE_ALL_SPLITS = ["low", "medium", "high", "top"]

lst_polymath_japanese_tasks = []
for split in POLYMATH_JAPANESE_ALL_SPLITS:
    _name = f"polymath_japanese:{split}"
    _task = LightevalTaskConfig(
        name=_name,
        suite=["swallow"],
        prompt_function=polymath_japanese_prompt_fn,
        hf_repo="Qwen/PolyMath",
        hf_subset="ja",
        hf_avail_splits=POLYMATH_JAPANESE_ALL_SPLITS,
        evaluation_splits=[split],
        few_shots_split=None,
        few_shots_select=None,
        metric=[latex_gold_metric],
        version=1,
    )
    lst_polymath_japanese_tasks.append(_task)

# Pass@K and Maj@K variant
lst_polymath_japanese_passk_majk = []
for polymath_task in lst_polymath_japanese_tasks:
    for num_samples in [8, 16, 32, 64, 128, 256]:
        lst_k = powers_of_two_up_to_n(num_samples)
        _lst_pass_at_k_metrics = create_passk_metrics(
            base_metric=latex_gold_metric,
            k_values=lst_k,
            num_samples=num_samples,
        )
        _lst_maj_at_k_metrics = create_majk_metrics(
            base_metric=latex_gold_metric,
            k_values=lst_k,
            num_samples=num_samples,
        )

        task_config = deepcopy(polymath_task)
        split = polymath_task.name.split(":")[1]
        task_config.name = f"polymath_japanese_N{num_samples}:{split}"
        task_config.metric = _lst_pass_at_k_metrics + _lst_maj_at_k_metrics
        lst_polymath_japanese_passk_majk.append(task_config)