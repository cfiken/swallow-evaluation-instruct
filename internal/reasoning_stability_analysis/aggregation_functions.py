
DUMMY_RESULT = {
    'num_responses': float('nan'),
    'num_non_closed_reasoning': float('nan'),
    'num_closed_reasoning': float('nan'),
    'reasoning_failure_ratio': float('nan'),
    'performance_in_completion': float('nan'),
    'performance': float('nan')
}


def extractive_match_metric(df_details, reasoning_starter: str) -> dict:
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
        
        if record["predictions"][0].startswith(reasoning_starter):
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

def ifeval_metric(df_details, reasoning_starter: str) -> dict:
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
        
        if record["predictions"][0].startswith(reasoning_starter):
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

def pass_at_k_metric(df_details, reasoning_starter: str) -> dict:
    """Pass@K Metric Benchmarks
    performance_in_completion is defined as the conditional average on the any one of the K responses correctly completed.
    """
    num_examples = 0
    num_instructions = 0
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    score_in_closed_reasoning = 0
    score_overall = 0
    for record in df_details.to_dict(orient="records"):
        dict_metrics = record["metrics"]
        if "humaneval_pass@1:10" in dict_metrics:
            score = dict_metrics["humaneval_pass@1:10"]
        elif "jhumaneval_pass@1:10" in dict_metrics:
            score = dict_metrics["jhumaneval_pass@1:10"]
        else:
            raise ValueError("pass@k metric not found in the record metrics.")
        score_overall += score
        lst_responses = list(record["predictions"])
        lst_is_non_closed_reasoning = [response.startswith(reasoning_starter) for response in lst_responses]
        num_non_closed_reasoning += sum(lst_is_non_closed_reasoning)
        num_closed_reasoning += len(lst_is_non_closed_reasoning) - sum(lst_is_non_closed_reasoning)
        num_examples += len(lst_responses)
        if not all(lst_is_non_closed_reasoning):
            score_in_closed_reasoning += score
        num_instructions += 1

    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "reasoning_failure_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_completion": score_in_closed_reasoning / num_instructions,
        "performance": score_overall / num_instructions
    }

    return dict_results


def bleu_metric(df_details, reasoning_starter: str) -> dict:
    """BLEU Metric Benchmarks

    BLEU metrics doesn't support performance_in_has_answer.
    """
    records = list(df_details.to_dict(orient="records"))
    num_examples = len(records)
    
    num_non_closed_reasoning = 0
    num_closed_reasoning = 0
    for record in df_details.to_dict(orient="records"):
        
        if record["predictions"][0].startswith(reasoning_starter):
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


def mt_bench_metric(df_details, reasoning_starter: str) -> dict:
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
            
            if response.startswith(reasoning_starter):
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