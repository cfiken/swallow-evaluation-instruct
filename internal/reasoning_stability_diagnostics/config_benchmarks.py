from aggregation_functions import extractive_match_metric, mt_bench_metric, ifeval_metric, pass_at_k_metric, bleu_metric_ja, bleu_metric_en, extractive_match_pass_at_k_metric

BENCHMARKS = {
    "swallow|japanese_mt_bench|0": {
        "analysis_function": mt_bench_metric,
        "file_pattern": "details_swallow|japanese_mt_bench|0_*.parquet"
    },
    "swallow|mifeval_ja|0": {
        "analysis_function": ifeval_metric,
        "file_pattern": "details_swallow|mifeval_ja|0_*.parquet"
    },
    "swallow|jamcqa|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|jamcqa|0_*.parquet"
    },
    "swallow|wmt20:en-ja|0": {
        "analysis_function": bleu_metric_ja,
        "file_pattern": "details_swallow|wmt20:en-ja|0_*.parquet"
    },
    "swallow|wmt20:ja-en|0": {
        "analysis_function": bleu_metric_en,
        "file_pattern": "details_swallow|wmt20:ja-en|0_*.parquet"
    },
    "swallow|swallow_gpqa_ja|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|swallow_gpqa_ja|0_*.parquet"
    },
    "swallow|math_100_japanese|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|math_100_japanese|0_*.parquet"
    },
    "swallow|mmlu_prox_japanese|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|mmlu_prox_japanese:*|0_*.parquet"
    },
    "swallow|swallow_jhumaneval|0": {
        "analysis_function": pass_at_k_metric,
        "file_pattern": "details_swallow|swallow_jhumaneval|0_*.parquet"
    },
    "swallow|hellaswag|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|hellaswag|0_*.parquet"
    },
    "swallow|mmlu_pro_english|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|mmlu_pro_english:*|0_*.parquet"
    },
    "swallow|gpqa:diamond|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|gpqa:diamond|0_*.parquet"
    },
    "swallow|math_500|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|math_500|0_*.parquet"
    },        
    "swallow|aime|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|aime:*|0_*.parquet"
    },
    "swallow|lcb:codegeneration_v5_v6|0": {
        "analysis_function": pass_at_k_metric,
        "file_pattern": "details_swallow|lcb:codegeneration_v5_v6|0_*.parquet"
    },
    "swallow|english_mt_bench|0": {
        "analysis_function": mt_bench_metric,
        "file_pattern": "details_swallow|english_mt_bench|0_*.parquet"
    },
}

EXTENDED_BENCHMARKS = {
    # "swallow|jemhopqa_cot|0": {
    #     "analysis_function": extractive_match_metric,
    #     "file_pattern": "details_swallow|jemhopqa_cot|0_*.parquet"
    # },    
    # "swallow|swallow_jmmlu|0": {
    #     "analysis_function": extractive_match_metric,
    #     "file_pattern": "details_swallow|swallow_jmmlu:*|0_*.parquet"
    # },    
    # "swallow|humaneval|0": {
    #     "analysis_function": pass_at_k_metric,
    #     "file_pattern": "details_swallow|humaneval|0_*.parquet"
    # },
    # "swallow|humanevalplus|0": {
    #     "analysis_function": pass_at_k_metric,
    #     "file_pattern": "details_swallow|humanevalplus|0_*.parquet"
    # },    
    "swallow|math_100_japanese_N16|0": {
        "analysis_function": extractive_match_pass_at_k_metric,
        "file_pattern": "details_swallow|math_100_japanese_N16|0_*.parquet"
    },    
    "swallow|gpqa_N16:diamond|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|gpqa:diamond|0_*.parquet"
    },
    "swallow|aime_N16|0": {
        "analysis_function": extractive_match_metric,
        "file_pattern": "details_swallow|aime_N16:*|0_*.parquet"
    },    
}
