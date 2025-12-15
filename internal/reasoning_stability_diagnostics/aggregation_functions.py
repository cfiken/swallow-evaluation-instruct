from typing import Optional
import math
import json
from copy import deepcopy

from utils import most_frequent_char_ngram, normalize_multiple_extractions
from bleu_scorer import bleu_score
from refusal_detector import is_refusal

def safe_divide(numerator: float, denominator: float, default: float = float('nan')) -> float:
    """安全な除算を行う。分母が0の場合はデフォルト値を返す"""
    if denominator == 0:
        return default
    return numerator / denominator

def is_refusal_fast(response: str, first_n_chars: int = 1000) -> bool:
    """拒否応答かどうかを高速に判定する簡易版"""
    if not response:
        return False
    text = response[:first_n_chars]
    return is_refusal(text)

DUMMY_RESULT = {
    'num_responses': float('nan'),
    'num_non_closed_reasoning': float('nan'),
    'num_closed_reasoning': float('nan'),
    'reasoning_failure_ratio': float('nan'),
    'performance_in_completion': float('nan'),
    'performance': float('nan'),
    "performance_delta": float('nan'),
    "refusal_ratio": float('nan'),
    "reserved_2": float('nan'),
    "reserved_3": float('nan'),
    "reserved_4": float('nan'),
}

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
    num_refusal = 0
    for record in df_details.to_dict(orient="records"):
        is_correct += record["metrics"]["extractive_match"]
        
        if is_non_closed_reasoning(record["predictions"][0], reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
            num_non_closed_reasoning += 1
        else:
            num_closed_reasoning += 1
            is_correct_in_closed_reasoning += record["metrics"]["extractive_match"]
        if is_refusal_fast(record["predictions"][0]):
            num_refusal += 1
    
    performance_in_completion = safe_divide(is_correct_in_closed_reasoning, num_closed_reasoning)
    performance = safe_divide(is_correct, num_examples)
    refusal_ratio = safe_divide(num_refusal, num_examples)
    
    dict_results = deepcopy(DUMMY_RESULT)
    dict_results.update({
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": performance_in_completion,
        "performance": performance,
        "performance_delta": performance_in_completion - performance,
        "refusal_ratio": refusal_ratio,
    })

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
    num_refusal = 0
    for record in df_details.to_dict(orient="records"):
        score = 1 if record["metrics"]["inst_level_strict_acc"][0] else 0
        is_correct += score
        
        if is_non_closed_reasoning(record["predictions"][0], reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
            num_non_closed_reasoning += 1
        else:
            num_closed_reasoning += 1
            is_correct_in_closed_reasoning += score
        if is_refusal_fast(record["predictions"][0]):
            num_refusal += 1
    
    performance_in_completion = is_correct_in_closed_reasoning / num_closed_reasoning
    performance = is_correct / num_examples
    refusal_ratio = num_refusal / num_examples
    
    dict_results = deepcopy(DUMMY_RESULT)
    dict_results.update({
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": performance_in_completion,
        "performance": performance,
        "performance_delta": performance_in_completion - performance,
        "refusal_ratio": refusal_ratio,
    })

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
    num_refusal = 0
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
            if is_refusal_fast(response):
                num_refusal += 1
        
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

    performance_in_completion = score_in_closed_reasoning / num_instructions
    performance = score_overall / num_instructions
    refusal_ratio = num_refusal / num_examples
    
    dict_results = deepcopy(DUMMY_RESULT)
    dict_results.update({
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": performance_in_completion,
        "performance": performance,
        "performance_delta": performance_in_completion - performance,
        "refusal_ratio": refusal_ratio,
    })
    
    original_performance = original_score_overall / num_instructions
    assert math.isclose(performance, original_performance, abs_tol=1e-2), "Pass@K performance calculation mismatch."

    return dict_results

def extractive_match_pass_at_k_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
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
    num_refusal = 0
    lst_metric_name_candidates = ["extractive_match_pass@1:8", "extractive_match_pass@1:16", "extractive_match_pass@1:32"]
    for record in df_details.to_dict(orient="records"):
        # instruction-level Pass@K score lookup
        dict_metrics = record["metrics"]
        for metric_name in lst_metric_name_candidates:
            if metric_name in dict_metrics:
                score_i = dict_metrics[metric_name]
                break
        else:
            raise ValueError("pass@k metric not found in the record metrics.")
        
        # response-level predictions
        lst_predictions = record["specifics"]["extracted_predictions"]
        # we always have two predictions for each instruction
        # e.g. ["27", "27", "81", "81", ...]
        lst_tup_predictions = list(zip(lst_predictions[0::2], lst_predictions[1::2]))
        # prediction frequency and correctness
        # e.g. [{'answer': '2187', 'count': 2, 'is_correct': False},
        # {'answer': '27', 'count': 1, 'is_correct': False},
        # {'answer': '81', 'count': 12, 'is_correct': True},
        # {'answer': '94', 'count': 1, 'is_correct': False}]
        lst_dict_predictions_freq = record["specifics"]["extracted_predictions_freq"]
        dict_answers = {item['answer']:item["is_correct"] for item in lst_dict_predictions_freq}
        
        # responses
        lst_responses = list(record["predictions"])
        assert len(lst_responses) == len(lst_tup_predictions), "Length mismatch between responses and extracted predictions."
        
        num_examples_i = 0
        passed_i = 0
        passed_in_completion_i = 0
        num_non_closed_reasoning_i = 0
        num_closed_reasoning_i = 0
        for response, tup_prediction in zip(lst_responses, lst_tup_predictions):
            lookup_key = normalize_multiple_extractions(tup_prediction)
            _passed = 1 if dict_answers[lookup_key] else 0
            
            if is_non_closed_reasoning(response, reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
                num_non_closed_reasoning_i += 1
            else:
                num_closed_reasoning_i += 1
                passed_in_completion_i += _passed
            if is_refusal_fast(response):
                num_refusal += 1
        
            num_examples_i += 1
            passed_i += _passed
        
        pass_at_1_i = _calculate_pass_at_k(n=num_examples_i, c=passed_i, k=1)
        pass_at_1_in_closed_reasoning_i = _calculate_pass_at_k(n=num_closed_reasoning_i, c=passed_in_completion_i, k=1)
        
        # sanity check
        if not math.isclose(pass_at_1_i, score_i, abs_tol=1e-3):
            # debugging output
            print(pass_at_1_i, score_i, num_examples_i, passed_i)
            print(lst_dict_predictions_freq)
            print(lst_predictions)
            print("======")
        
        num_non_closed_reasoning += num_non_closed_reasoning_i
        num_closed_reasoning += num_closed_reasoning_i
        num_examples += num_examples_i
        score_overall += pass_at_1_i
        score_in_closed_reasoning += pass_at_1_in_closed_reasoning_i
        num_instructions += 1
        original_score_overall += score_i

    performance_in_completion = score_in_closed_reasoning / num_instructions
    performance = score_overall / num_instructions
    refusal_ratio = num_refusal / num_examples
    
    dict_results = deepcopy(DUMMY_RESULT)
    dict_results.update({
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": performance_in_completion,
        "performance": performance,
        "performance_delta": performance_in_completion - performance,
        "refusal_ratio": refusal_ratio,
    })
    
    original_performance = original_score_overall / num_instructions
    assert math.isclose(performance, original_performance, abs_tol=1e-2), "Pass@K performance calculation mismatch."

    return dict_results


def _bleu_metric_common(df_details, reasoning_starter: Optional[str], repetition_ngram: int, top_ngram_freq_repetition_threshold: int, trg_lang: str) -> dict:
    """BLEU Metric Benchmarks (共通実装)

    BLEU metrics doesn't support performance_in_has_answer.
    
    Args:
        df_details: データフレーム
        reasoning_starter: 推論開始マーカー
        repetition_ngram: N-gramのサイズ
        top_ngram_freq_repetition_threshold: 最頻N-gramの閾値
        trg_lang: ターゲット言語 ("ja" または "en")
    """
    records = list(df_details.to_dict(orient="records"))
    num_examples = len(records)
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    num_refusal = 0
    predictions = []
    golds = []
    predictions_in_closed_reasoning = []
    golds_in_closed_reasoning = []
    for record in df_details.to_dict(orient="records"):
        
        # prediction and gold are tokenized sentences used for BLEU calculation
        prediction = record["metrics"]["bleu"]["preds"][0]
        gold = record["metrics"]["bleu"]["golds"][0]
        predictions.append(prediction)
        golds.append(gold)
        
        if is_non_closed_reasoning(record["predictions"][0], reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold):
            num_non_closed_reasoning += 1
        else:
            num_closed_reasoning += 1
            predictions_in_closed_reasoning.append(prediction)
            golds_in_closed_reasoning.append(gold)
        if is_refusal_fast(record["predictions"][0]):
            num_refusal += 1
            
    bleu_overall = bleu_score(predictions=predictions, golds=golds, trg_lang=trg_lang)
    bleu_in_closed_reasoning = bleu_score(predictions=predictions_in_closed_reasoning, golds=golds_in_closed_reasoning, trg_lang=trg_lang)
    refusal_ratio = num_refusal / num_examples
    
    dict_results = deepcopy(DUMMY_RESULT)
    dict_results.update({
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": bleu_in_closed_reasoning,
        "performance": bleu_overall,
        "performance_delta": bleu_in_closed_reasoning - bleu_overall,
        "refusal_ratio": refusal_ratio,
    })

    return dict_results


def bleu_metric_ja(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """BLEU Metric Benchmarks xx-to-Japanese
    """
    # sentences in lighteval details are already tokenized for Japanese BLEU calculation, thus trg_lang=""
    return _bleu_metric_common(df_details, reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold, trg_lang="")


def bleu_metric_en(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """BLEU Metric Benchmarks xx-to-English
    """
    return _bleu_metric_common(df_details, reasoning_starter, repetition_ngram, top_ngram_freq_repetition_threshold, trg_lang="en")


def mt_bench_metric(df_details, reasoning_starter: Optional[str], repetition_ngram: int = 50, top_ngram_freq_repetition_threshold: int = 10) -> dict:
    """MT-Bench Metric Benchmarks
    """
    records = list(df_details.to_dict(orient="records"))
    
    num_examples = 0
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    score_in_closed_reasoning = 0
    score_overall = 0
    num_refusal = 0
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
            if is_refusal_fast(response):
                num_refusal += 1
            score_overall += score
    
    performance_in_completion = score_in_closed_reasoning / num_closed_reasoning / 10
    performance = score_overall / num_examples / 10
    refusal_ratio = num_refusal / num_examples
    
    dict_results = deepcopy(DUMMY_RESULT)
    dict_results.update({
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": performance_in_completion,
        "performance": performance,
        "performance_delta": performance_in_completion - performance,
        "refusal_ratio": refusal_ratio,
    })

    return dict_results
