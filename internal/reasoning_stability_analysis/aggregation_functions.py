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
            is_correct_in_closed_reasoning += is_correct
    
    dict_results = {
        "num_responses": num_examples,
        "num_non_closed_reasoning": num_non_closed_reasoning,
        "num_closed_reasoning": num_closed_reasoning,
        "no_answer_ratio": num_non_closed_reasoning / num_examples,
        "performance_in_has_answer": is_correct_in_closed_reasoning / num_closed_reasoning,
        "performance_overall": is_correct / num_examples
    }

    return dict_results
