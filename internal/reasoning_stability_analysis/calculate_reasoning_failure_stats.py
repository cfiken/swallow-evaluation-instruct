#!/usr/bin/env python3
"""推論過程の安定性分析プログラム

このスクリプトは、指定されたモデルとタスクについて、
HuggingFaceからデータセットをダウンロードし、
推論過程の安定性を分析して結果を出力します。
"""

import argparse
import json
import os
import sys
from typing import Optional

import pandas as pd
from datasets import load_dataset

from config_benchmarks import BENCHMARKS


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
        type=str,
        default='<think>',
        help='推論開始タグ (デフォルト: <think>)'
    )
    parser.add_argument(
        '--provider',
        type=str,
        default='hosted_vllm',
        help='プロバイダー名 (デフォルト: hosted_vllm)'
    )
    parser.add_argument(
        '--task_ids',
        type=str,
        nargs='*',
        default=None,
        help='分析対象のタスクID（指定なしの場合は全タスク）'
    )
    return parser.parse_args()


def build_hf_dataset_id(model_id: str, hf_organization: str, provider: str) -> str:
    """HuggingFaceのデータセットIDを構築する
    
    Args:
        model_id: モデルID
        hf_organization: HuggingFace組織名
        provider: プロバイダー名
        
    Returns:
        構築されたデータセットID
    """
    model_name = f"{provider}/{model_id}"
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
    reasoning_starter: str,
    model_id: str
) -> dict:
    """タスクデータをロードして分析する
    
    Args:
        hf_dataset_id: HuggingFaceデータセットID
        task_id: タスクID
        reasoning_starter: 推論開始タグ
        model_id: モデルID
        
    Returns:
        分析結果の辞書
        
    Raises:
        KeyError: タスクIDがBENCHMARKSに存在しない場合
        Exception: データセットのロードや分析に失敗した場合
    """
    # タスクIDがBENCHMARKSに存在するか確認
    if task_id not in BENCHMARKS:
        raise KeyError(f"タスクID '{task_id}' はBENCHMARKSに登録されていません")
    
    # データセットのサブセット名を構築
    subset = task_id.replace("|", "_").replace(":", "_")
    
    # データセットをロード
    dataset = load_dataset(hf_dataset_id, name=subset, split="latest")
    df_details = dataset.to_pandas()
    
    # 分析関数を取得して実行
    analysis_function = BENCHMARKS[task_id]
    result = analysis_function(df_details, reasoning_starter)
    
    # model_idを結果に追加
    result['model_id'] = model_id
    
    return result


def save_results_json(results: dict, hf_dataset_id: str) -> str:
    """結果をJSONファイルに保存する
    
    Args:
        results: 分析結果の辞書
        hf_dataset_id: データセットID（ファイル名に使用）
        
    Returns:
        保存したファイルのパス
    """
    output_path = f'results/{hf_dataset_id}.json'
    # 親ディレクトリを作成（スラッシュが含まれる場合はサブディレクトリも作成）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    return output_path


def save_results_csv(results: dict, hf_dataset_id: str) -> str:
    """結果をCSV形式で保存する
    
    Args:
        results: 分析結果の辞書
        hf_dataset_id: データセットID（ファイル名に使用）
        
    Returns:
        保存したファイルのパス
    """
    output_path = f'results/{hf_dataset_id}.csv'
    # 親ディレクトリを作成（スラッシュが含まれる場合はサブディレクトリも作成）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # タスクIDをインデックスから列に移動
    df = pd.DataFrame(results).T
    df.index.name = 'task_id'
    df.reset_index(inplace=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    return output_path


def main():
    """メイン処理"""
    # コマンドライン引数のパース
    args = parse_args()
    
    # HuggingFaceデータセットIDの構築
    hf_dataset_id = build_hf_dataset_id(
        args.model_id,
        args.hf_organization,
        args.provider
    )
    
    print(f"データセットID: {hf_dataset_id}", file=sys.stderr)
    print(f"モデルID: {args.model_id}", file=sys.stderr)
    
    # 対象タスクIDの取得
    task_ids = get_task_ids(args.task_ids)
    print(f"対象タスク数: {len(task_ids)}", file=sys.stderr)
    
    # 各タスクの分析
    results = {}
    for i, task_id in enumerate(task_ids, 1):
        print(f"\n[{i}/{len(task_ids)}] タスク {task_id} を処理中...", file=sys.stderr)
        try:
            result = load_and_analyze_task(
                hf_dataset_id,
                task_id,
                args.reasoning_starter,
                args.model_id
            )
            results[task_id] = result
            print(f"  ✓ 完了", file=sys.stderr)
        except Exception as e:
            print(f"  ✗ 警告: タスク {task_id} の処理に失敗しました: {e}", file=sys.stderr)
            continue
    
    # 結果が1つもない場合はエラー
    if not results:
        print("\nエラー: すべてのタスクの処理に失敗しました", file=sys.stderr)
        sys.exit(1)
    
    print(f"\n成功したタスク: {len(results)}/{len(task_ids)}", file=sys.stderr)
    
    # 標準出力（簡素版）
    print(f"タスク別の無回答率:")
    print(f"\n{args.model_id}")
    for task_id, result in results.items():
        no_answer_ratio = result.get('no_answer_ratio', 'N/A')
        print(f"{task_id}\t{no_answer_ratio}")
    
    # JSONファイル出力
    json_path = save_results_json(results, hf_dataset_id)
    print(f"\nJSONファイル保存: {json_path}", file=sys.stderr)
    
    # CSV出力
    csv_path = save_results_csv(results, hf_dataset_id)
    print(f"CSVファイル保存: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
