# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Pass@K metric utilities for creating Pass@K versions of existing SampleLevelMetric.
This module provides utilities to convert any SampleLevelMetric into a Pass@K metric 
that can be used across different benchmarks.
"""

from typing import List
import numpy as np

from lighteval.metrics.utils.metric_utils import (
    MetricUseCase,
    MetricCategory,
    SampleLevelMetric,
)
from lighteval.tasks.requests import Doc


def estimate_pass_at_k(num_samples: int, num_correct: int, k: int) -> float:
    """Estimates pass@k for a single problem."""
    if num_samples - num_correct < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))

def powers_of_two_up_to_n(N):
    """Generates all powers of two up to a given number N.

    Args:
        N (int): The upper limit (inclusive) for generating powers of two.

    Returns:
        List[int]: A list of all powers of two up to N.
    """
    result = []
    value = 1
    while value <= N:
        result.append(value)
        value *= 2
    if result[-1] != N:
        result.append(N)
    return result

def create_passk_metric_fn(base_metric: SampleLevelMetric, k: int) -> callable:
    """
    SampleLevelMetricをPass@K対応に変換する関数を作成します。
    
    Args:
        base_metric: 基となるSampleLevelMetric
        k: Pass@Kのk値
    
    Returns:
        Pass@K対応のメトリクス関数
    """
    def passk_metric_fn(predictions: List[str], formatted_doc: Doc, **kwargs) -> float:
        """
        Pass@K指標を計算するメトリクス関数
        
        Args:
            predictions: 予測結果のリスト（K個以上）
            formatted_doc: フォーマット済みドキュメント
            **kwargs: メトリクス関数に渡す追加引数
        
        Returns:
            Pass@K値
        """
        if len(predictions) < k:
            raise ValueError(f"Number of predictions ({len(predictions)}) is less than k ({k}) for Pass@{k}")
        
        # 各予測に対して基となるメトリクスを計算
        scores = []
        golds = formatted_doc.get_golds()
        
        for pred in predictions:
            # 基となるメトリクス関数を呼び出し
            score = base_metric.sample_level_fn(golds=golds, predictions=[pred], formatted_doc=formatted_doc, **kwargs)
            scores.append(int(score))
        
        # Pass@K計算: estimate_pass_at_kを使用
        num_samples = len(predictions)
        num_correct = sum(scores)
        
        # 単一サンプルに対するPass@K計算
        pass_at_k_score = estimate_pass_at_k(
            num_samples=num_samples, 
            num_correct=num_correct, 
            k=k
        )
        
        return pass_at_k_score
    
    return passk_metric_fn


def create_passk_metrics(base_metric: SampleLevelMetric, k_values: List[int], num_samples: int) -> List[SampleLevelMetric]:
    """
    SampleLevelMetricから複数のK値に対してPass@Kメトリクスを作成します。
    
    Args:
        base_metric: 基となるSampleLevelMetric
        k_values: Pass@KのK値のリスト
        num_samples: サンプリング数
    
    Returns:
        Pass@Kメトリクスのリスト
    """
    assert base_metric.use_case == MetricUseCase.ACCURACY, "Base metric must be an accuracy-type metric."

    metrics = []
    for k in k_values:
        # Pass@K用のメトリクス関数を作成
        passk_fn = create_passk_metric_fn(base_metric, k)
        
        # メトリクス名を生成（基のメトリクス名にpass@kを追加）
        base_name = base_metric.metric_name
        metric_name = f"{base_name}_pass@{k}:{num_samples}"
        
        # SampleLevelMetricを作成
        metric = SampleLevelMetric(
            metric_name=metric_name,
            category=MetricCategory.GENERATIVE_SAMPLING,
            use_case=base_metric.use_case,
            higher_is_better=True,
            sample_level_fn=passk_fn,
            corpus_level_fn=base_metric.corpus_level_fn,
        )
        
        metrics.append(metric)
    
    return metrics
