from typing import Optional
import math
import json

from utils import most_frequent_char_ngram


def is_non_closed_reasoning(
    reasoning_content: str | None,
    reasoning_starter: Optional[str] = None,
    repetition_ngram: int = 50,
    top_ngram_freq_repetition_threshold: int = 10,
    num_chars_threshold: int = 20000
) -> bool:
    """レスポンスがnon-closed reasoning（推論が閉じていない）かどうかを判定
    
    Args:
        response: 判定対象のレスポンステキスト
        reasoning_starter: 推論開始マーカー
        repetition_ngram: N-gram のサイズ (デフォルト: 50)
        top_ngram_freq_repetition_threshold: 最頻N-gramの閾値 (デフォルト: 10)
        
    Returns:
        True if non-closed reasoning, False otherwise
    """
    # reasoning_contentがNoneの場合はTrueを返す
    if reasoning_content is None:
        return True
    
    if reasoning_starter is not None:
        # 条件1: reasoning_starterで始まる
        return reasoning_content.startswith(reasoning_starter)
    else:
        # 条件2: 最頻N-gramのfrequencyが閾値以上
        ngram_stats = most_frequent_char_ngram(reasoning_content, n=repetition_ngram)
        if (ngram_stats["frequency"] >= top_ngram_freq_repetition_threshold) and len(reasoning_content) >= num_chars_threshold:
            return True    
        return False


DUMMY_RESULT = {
    'num_responses': float('nan'),
    'num_non_closed_reasoning': float('nan'),
    'num_closed_reasoning': float('nan'),
    'reasoning_failure_ratio': float('nan'),
    'performance_in_completion': float('nan'),
    'performance': float('nan')
}


def extractive_match_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """Extractive Match Metric Benchmarks

    Extractive match metrics evaluate how well the aggregated
    reasoning steps match the expected answers in benchmark datasets.
    This function computes benchmarks for these metrics.
    """
    records = list(df_details.to_dict(orient="records"))
    num_examples = len(records)
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    is_correct_in_closed_reasoning = 0
    is_correct = 0
    for record in df_details.to_dict(orient="records"):
        is_correct += record["metrics"]["extractive_match"]
        
        if is_non_closed_reasoning(record["predictions"][0], reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
            num_non_closed_reasoning += 1
        else:
            num_closed_reasoning += 1
            is_correct_in_closed_reasoning += record["metrics"]["extractive_match"]
    
    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": is_correct_in_closed_reasoning / num_closed_reasoning,
        "performance": is_correct / num_examples
    }

    return dict_results

def ifeval_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """IFEval Metric Benchmarks
    """
    records = list(df_details.to_dict(orient="records"))
    num_examples = len(records)
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    is_correct_in_closed_reasoning = 0
    is_correct = 0
    for record in df_details.to_dict(orient="records"):
        score = 1 if record["metrics"]["inst_level_strict_acc"][0] else 0
        is_correct += score
        
        if is_non_closed_reasoning(record["predictions"][0], reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
            num_non_closed_reasoning += 1
        else:
            num_closed_reasoning += 1
            is_correct_in_closed_reasoning += score
    
    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": is_correct_in_closed_reasoning / num_closed_reasoning,
        "performance": is_correct / num_examples
    }

    return dict_results

def _calculate_pass_at_k(n: int, c: int, k: int) -> float:
    """Calculates 1 - comb(n - c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.prod(1.0 - k / i for i in range(n - c + 1, n + 1))

def pass_at_k_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """Pass@K Metric Benchmarks
    performance_in_completion is defined as the conditional average on the any one of the K responses correctly completed.
    """
    num_examples = 0
    num_instructions = 0
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    score_in_closed_reasoning = 0
    score_overall = 0
    original_score_overall = 0
    lst_metric_name_candidates = ["humaneval_pass@1:10", "jhumaneval_pass@1:10", "codegen_pass@1:10"]
    for record in df_details.to_dict(orient="records"):
        # instruction-level Pass@K score lookup
        dict_metrics = record["metrics"]
        for metric_name in lst_metric_name_candidates:
            if metric_name in dict_metrics:
                score_i = dict_metrics[metric_name]
                break
        else:
            raise ValueError("pass@k metric not found in the record metrics.")
        
        # response-level unit-test results
        lst_lst_unit_test_results = json.loads(record["specifics"]["results"])
        
        # responses
        lst_responses = list(record["predictions"])
        num_examples_i = 0
        passed_i = 0
        passed_in_completion_i = 0
        num_non_closed_reasoning_i = 0
        num_closed_reasoning_i = 0
        for response, lst_unit_results in zip(lst_responses, lst_lst_unit_test_results):
            _passed = 1 if all([result == True for result in lst_unit_results]) else 0
            if is_non_closed_reasoning(response, reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
                num_non_closed_reasoning_i += 1
            else:
                num_closed_reasoning_i += 1
                passed_in_completion_i += _passed
        
            num_examples_i += 1
            passed_i += _passed
        
        pass_at_1_i = _calculate_pass_at_k(n=num_examples_i, c=passed_i, k=1)
        pass_at_1_in_closed_reasoning_i = _calculate_pass_at_k(n=num_closed_reasoning_i, c=passed_in_completion_i, k=1)
        
        # sanity check
        if not math.isclose(pass_at_1_i, score_i, abs_tol=1e-3):
            # debugging output
            print(pass_at_1_i, score_i, num_examples_i, passed_i)            
        
        num_non_closed_reasoning += num_non_closed_reasoning_i
        num_closed_reasoning += num_closed_reasoning_i
        num_examples += num_examples_i
        score_overall += pass_at_1_i
        score_in_closed_reasoning += pass_at_1_in_closed_reasoning_i
        num_instructions += 1
        original_score_overall += score_i

    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": score_in_closed_reasoning / num_instructions,
        "performance": score_overall / num_instructions
    }
    
    original_performance = original_score_overall / num_instructions
    assert math.isclose(score_overall / num_instructions, original_performance, abs_tol=1e-2), "Pass@K performance calculation mismatch."

    return dict_results


def bleu_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """BLEU Metric Benchmarks

    BLEU metrics doesn't support performance_in_has_answer.
    """
    records = list(df_details.to_dict(orient="records"))
    num_examples = len(records)
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    for record in df_details.to_dict(orient="records"):
        
        if is_non_closed_reasoning(record["predictions"][0], reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
            num_non_closed_reasoning += 1
        else:
            num_closed_reasoning += 1
    
    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": float("nan"),
        "performance": float("nan")
    }

    return dict_results


def mt_bench_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """MT-Bench Metric Benchmarks
    """
    records = list(df_details.to_dict(orient="records"))
    
    num_examples = 0
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    score_in_closed_reasoning = 0
    score_overall = 0
    for record in df_details.to_dict(orient="records"):
        lst_1st_turn_scores = list(record["metrics"]["judge_score_overall_turn_1"])
        lst_2nd_turn_scores = list(record["metrics"]["judge_score_overall_turn_2"])
        lst_1st_turn_responses = list(record["predictions"][0])
        lst_2nd_turn_responses = list(record["predictions"][1])

        lst_scores = lst_1st_turn_scores + lst_2nd_turn_scores
        lst_responses = lst_1st_turn_responses + lst_2nd_turn_responses
        assert len(lst_scores) == len(lst_responses), "Length mismatch between scores and responses in MT-Bench metric."
        for score, response in zip(lst_scores, lst_responses):
            num_examples += 1
            
            if is_non_closed_reasoning(response, reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold, 
                                       num_chars_threshold=8000):
                num_non_closed_reasoning += 1
            else:
                num_closed_reasoning += 1
                score_in_closed_reasoning += score
            score_overall += score
    
    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": score_in_closed_reasoning / num_closed_reasoning / 10,
        "performance": score_overall / num_examples / 10,
    }

    return dict_results
