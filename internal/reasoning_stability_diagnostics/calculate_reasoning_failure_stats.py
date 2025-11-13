#!/usr/bin/env python3
"""推論過程の安定性分析プログラム

このスクリプトは、指定されたモデルとタスクについて、
HuggingFaceからデータセットをダウンロードし、
推論過程の安定性を分析して結果を出力します。
"""

import argparse
import copy
import json
import os
import sys
from typing import Optional

import pandas as pd
from datasets import load_dataset

from config_benchmarks import BENCHMARKS
from aggregation_functions import DUMMY_RESULT
from utils import load_parquet_from_local


def parse_args() -> argparse.Namespace:
    """コマンドライン引数をパースする"""
    parser = argparse.ArgumentParser(
        description='推論過程の安定性分析',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--model_id',
        type=str,
        required=True,
        help='モデルID (例: tokyotech-llm/Qwen3-Swallow-8B-v0.1-SFT-exp12-LR1.5E-5-iter0023000)'
    )
    parser.add_argument(
        '--hf_organization',
        type=str,
        default='tokyotech-llm',
        help='HuggingFace組織名 (デフォルト: tokyotech-llm)'
    )
    parser.add_argument(
        '--reasoning_starter',
        type=lambda x: None if x.lower() == 'none' else x,
        required=True,
        choices=["<think>", "<think_dummy>", "None"],
        help='推論開始タグ (例: <think>)。gpt-oss系列の場合は"<think_dummy>"を指定してください'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default=None,
        help='LLMプロバイダー名（例：hosted_vllm, 省略可）'
    )
    parser.add_argument(
        '--task_ids',
        type=str,
        nargs='*',
        default=None,
        help='分析対象のタスクID（指定なしの場合は全タスク）'
    )
    parser.add_argument(
        '--lighteval-output-dir',
        type=str,
        default=None,
        help='ローカルストレージからParquetファイルを読み込む場合のディレクトリパス'
    )
    parser.add_argument(
        '--repetition-ngram',
        type=int,
        default=50,
        help='N-gram サイズ (デフォルト: 50)'
    )
    parser.add_argument(
        '--top-ngram-freq-repetition-threshold',
        type=int,
        default=10,
        help='最頻N-gram頻度の閾値 (デフォルト: 10)'
    )
    parser.add_argument(
        '--output-basename',
        type=str,
        default=None,
        help='出力ファイルのベース名（指定なしの場合はhf_dataset_idを使用）'
    )
    parser.add_argument(
        '--append',
        action='store_true',
        help='既存ファイルに追記する（CSVはヘッダーなし、JSONLは追記）'
    )
    return parser.parse_args()


def build_hf_dataset_id(model_id: str, hf_organization: str, provider: Optional[str]) -> str:
    """HuggingFaceのデータセットIDを構築する
    
    Args:
        model_id: モデルID
        hf_organization: HuggingFace組織名
        provider: プロバイダー名（Noneの場合はproviderなし）
        
    Returns:
        構築されたデータセットID
    """
    if provider:
        model_name = f"{provider}/{model_id}"
    else:
        model_name = model_id
    return f"{hf_organization}/details_{model_name.replace('/', '__')}_private"


def get_task_ids(specified_task_ids: Optional[list[str]]) -> list[str]:
    """分析対象のタスクIDリストを取得する
    
    Args:
        specified_task_ids: ユーザー指定のタスクID（Noneの場合は全タスク）
        
    Returns:
        タスクIDのリスト
    """
    if specified_task_ids is None:
        return list(BENCHMARKS.keys())
    return specified_task_ids


def load_and_analyze_task(
    hf_dataset_id: str,
    task_id: str,
    reasoning_starter: Optional[str],
    model_id: str,
    lighteval_output_dir: Optional[str] = None,
    provider: Optional[str] = None,
    repetition_ngram: int = 50,
    top_ngram_freq_repetition_threshold: int = 10
) -> dict:
    """タスクデータをロードして分析する
    
    Args:
        hf_dataset_id: HuggingFaceデータセットID
        task_id: タスクID
        reasoning_starter: 推論開始タグ（Noneの場合はN-gramのみでチェック）
        model_id: モデルID
        lighteval_output_dir: ローカルストレージのディレクトリ（Noneの場合はHFモード）
        provider: プロバイダー名（ローカルモード用）
        repetition_ngram: N-gram のサイズ
        top_ngram_freq_repetition_threshold: 最頻N-gramの閾値
        
    Returns:
        分析結果の辞書
        
    Raises:
        KeyError: タスクIDがBENCHMARKSに存在しない場合
        Exception: データセットのロードや分析に失敗した場合
    """
    # タスクIDがBENCHMARKSに存在するか確認
    if task_id not in BENCHMARKS:
        raise KeyError(f"タスクID '{task_id}' はBENCHMARKSに登録されていません")
    
    benchmark_config = BENCHMARKS[task_id]
    
    # データをロード
    if lighteval_output_dir is None:
        # HuggingFaceモード
        subset = task_id.replace("|", "_").replace(":", "_")
        dataset = load_dataset(hf_dataset_id, name=subset, split="latest")
        df_details = dataset.to_pandas()
    else:
        # ローカルストレージモード
        file_pattern = benchmark_config["file_pattern"]
        df_details = load_parquet_from_local(
            lighteval_output_dir,
            model_id,
            provider,
            task_id,
            file_pattern
        )
    
    # 分析関数を取得して実行
    analysis_function = benchmark_config["analysis_function"]
    result = analysis_function(
        df_details,
        reasoning_starter,
        repetition_ngram=repetition_ngram,
        top_ngram_freq_repetition_threshold=top_ngram_freq_repetition_threshold
    )
    
    # model_idを結果に追加
    result['model_id'] = model_id
    
    return result


def save_results_json(results: dict, output_basename: str, append: bool = False) -> str:
    """結果をJSONファイルに保存する
    
    Args:
        results: 分析結果の辞書
        output_basename: 出力ファイルのベース名
        append: 追記モードかどうか
        
    Returns:
        保存したファイルのパス
    """
    output_path = f'results/{output_basename}.json'
    # 親ディレクトリを作成（スラッシュが含まれる場合はサブディレクトリも作成）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 構造を model_id -> task_id -> metrics に変更
    # results は Dict[task_id, metrics + model_id] の形式
    restructured = {}
    for task_id, metrics_with_model in results.items():
        model_id = metrics_with_model.get('model_id', 'unknown')
        # model_id を除いたメトリクスを取得
        metrics = {k: v for k, v in metrics_with_model.items() if k != 'model_id'}
        
        # model_id -> task_id -> metrics の階層構造を作成
        if model_id not in restructured:
            restructured[model_id] = {}
        restructured[model_id][task_id] = metrics
    
    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        json.dump(restructured, f, ensure_ascii=False)
        f.write('\n')
    
    return output_path


def save_results_csv(results: dict, output_basename: str, append: bool = False) -> str:
    """結果をCSV形式で保存する
    
    Args:
        results: 分析結果の辞書
        output_basename: 出力ファイルのベース名
        append: 追記モードかどうか
        
    Returns:
        保存したファイルのパス
    """
    output_path = f'results/{output_basename}.csv'
    # 親ディレクトリを作成（スラッシュが含まれる場合はサブディレクトリも作成）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # タスクIDをインデックスから列に移動
    df = pd.DataFrame(results).T
    df.index.name = 'task_id'
    df.reset_index(inplace=True)
    
    # 列の順序を変更: model_id を左端, task_id を2番目
    other_cols = [col for col in df.columns if col not in ['model_id', 'task_id']]
    df = df[['model_id', 'task_id'] + other_cols]
    
    # appendモードの場合はヘッダーなしで追記
    mode = 'a' if append else 'w'
    if append and os.path.exists(output_path):
        header = False
    else:
        header = True
    df.to_csv(output_path, mode=mode, header=header, index=False, encoding='utf-8', na_rep='#N/A')
    
    return output_path


def save_results_csv_oneliner(results: dict, output_basename: str, append: bool = False) -> str:
    """結果をワンライナー形式のCSV（wide format）で保存する
    
    model_idをインデックス、task_id+メトリクス名をカラムとした形式で保存する。
    各モデルが1行で表現される。
    
    Args:
        results: 分析結果の辞書
        output_basename: 出力ファイルのベース名
        append: 追記モードかどうか
        
    Returns:
        保存したファイルのパス
    """
    output_path = f'results/{output_basename}.oneliner.csv'
    # 親ディレクトリを作成（スラッシュが含まれる場合はサブディレクトリも作成）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Long format DataFrameを作成
    df_long = pd.DataFrame(results).T
    df_long.index.name = 'task_id'
    df_long.reset_index(inplace=True)
    
    # model_idとtask_idを分離し、その他の列をメトリクスとして扱う
    metric_cols = [col for col in df_long.columns if col not in ['model_id', 'task_id']]
    
    # Wide formatに変換: model_idをインデックス、task_id_metricをカラムに
    df_wide = df_long.set_index(['model_id', 'task_id'])[metric_cols].unstack(level='task_id')
    
    # カラム名を task_id_metric の形式にフラット化
    df_wide.columns = [f'{task_id}_{metric}' for metric, task_id in df_wide.columns]
    
    # インデックスをリセットしてmodel_idを通常の列にする
    df_wide.reset_index(inplace=True)
    
    # BENCHMARKS.keys()の順序でカラムを並び替え
    # まず model_id 列を取得
    model_id_col = df_wide[['model_id']]
    
    # BENCHMARKSの順序でタスクIDを取得
    ordered_task_ids = list(BENCHMARKS.keys())
    
    # 各タスクについて、存在するメトリクスカラムを順序通りに追加
    ordered_cols = ['model_id']
    for task_id in ordered_task_ids:
        # このタスクに関連するカラム（task_id_metric形式）を探す
        task_cols = [col for col in df_wide.columns if col.startswith(f'{task_id}_')]
        ordered_cols.extend(task_cols)
    
    # 存在しないカラムを除外して並び替え
    available_cols = [col for col in ordered_cols if col in df_wide.columns]
    df_wide = df_wide[available_cols]
    
    # appendモード & ファイルが存在する場合はヘッダーなしで追記
    mode = 'a' if append else 'w'
    if append and os.path.exists(output_path):
        header = False
    else:
        header = True
    df_wide.to_csv(output_path, mode=mode, header=header, index=False, encoding='utf-8', na_rep='#N/A')
    
    return output_path


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_args()
    
    # モード判定
    lighteval_output_dir = getattr(args, 'lighteval_output_dir', None)
    is_local_mode = lighteval_output_dir is not None
    
    # ローカルモードの場合はhf_organizationをLOCALで上書き
    hf_organization = "LOCAL" if is_local_mode else args.hf_organization
    
    # HuggingFaceデータセットIDの構築
    hf_dataset_id = build_hf_dataset_id(
        args.model_id,
        hf_organization,
        args.provider
    )
    
    # モード情報を表示
    if is_local_mode:
        print(f"モード: local storage", file=sys.stderr)
        print(f"lighteval output dir: {lighteval_output_dir}", file=sys.stderr)
    else:
        print(f"モード: HuggingFace", file=sys.stderr)
        print(f"HF Dataset ID: {hf_dataset_id}", file=sys.stderr)
    
    print(f"Model ID: {args.model_id}", file=sys.stderr)
    print(f"LLM Provider: {args.provider}", file=sys.stderr)
    
    # 対象タスクIDの取得
    task_ids = get_task_ids(args.task_ids)
    print(f"対象タスク数: {len(task_ids)}", file=sys.stderr)
    
    # 各タスクの分析
    results = {}
    num_succeeded = 0
    for i, task_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}] タスク {task_id} を処理中...", file=sys.stderr)
        try:
            result = load_and_analyze_task(
                hf_dataset_id,
                task_id,
                args.reasoning_starter,
                args.model_id,
                lighteval_output_dir,
                args.provider if is_local_mode else None,
                args.repetition_ngram,
                args.top_ngram_freq_repetition_threshold
            )
            results[task_id] = result
            num_succeeded += 1
            print(f"  ✓ 完了", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ 警告: タスク {task_id} の処理に失敗しました: {e}", file=sys.stderr)
            results[task_id] = copy.deepcopy(DUMMY_RESULT)
            results[task_id]['model_id'] = args.model_id
    
    # 成功したタスクが1つもない場合は警告
    if num_succeeded == 0:
        print("\n警告: すべてのタスクの処理に失敗しました", file=sys.stderr)
    
    print(f"\n成功したタスク: {num_succeeded}/{len(task_ids)}", file=sys.stderr)
    
    # 標準出力（簡素版）
    print(f"タスク別の無回答率:")
    print(f"\n{args.model_id}")
    lst_header = ["task_id", "reasoning_failure_ratio", "performance_in_completion", "performance"]
    header = ",".join(lst_header)
    print(header)
    for task_id, result in results.items():
        lst_values = [result.get(metric_name, 'N/A') for metric_name in lst_header[1:]]
        str_values = ",".join(map(lambda v: f"{v:.3f}", lst_values))
        print(f"{task_id},{str_values}")

    # 出力ファイルのベース名を決定
    output_basename = args.output_basename if args.output_basename else hf_dataset_id
    
    # JSONファイル出力
    json_path = save_results_json(results, output_basename, args.append)
    print(f"\nJSONファイル保存: {json_path}", file=sys.stderr)
    
    # CSV出力
    csv_path = save_results_csv(results, output_basename, args.append)
    print(f"CSVファイル保存: {csv_path}", file=sys.stderr)
    
    # ワンライナーCSV出力
    csv_oneliner_path = save_results_csv_oneliner(results, output_basename, args.append)
    print(f"ワンライナーCSVファイル保存: {csv_oneliner_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
