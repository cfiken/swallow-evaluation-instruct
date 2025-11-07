from pathlib import Path
from typing import Optional
from collections import defaultdict
from fnmatch import fnmatch
import pandas as pd


def load_parquet_from_local(
    lighteval_output_dir: str,
    model_id: str,
    provider: Optional[str],
    task_id: str,
    file_pattern: str
) -> pd.DataFrame:
    """ローカルストレージからParquetファイルを読み込む
    
    Args:
        lighteval_output_dir: lighteval出力ディレクトリのパス
        model_id: モデルID
        provider: プロバイダー名（省略可）
        task_id: タスクID（エラーメッセージ用）
        file_pattern: ファイルパターン（ワイルドカード含む）
        
    Returns:
        結合されたDataFrame
        
    Raises:
        FileNotFoundError: ディレクトリまたはファイルが見つからない場合
    """
    # ベースディレクトリ構築
    if provider:
        base_path = Path(lighteval_output_dir) / "details" / provider / model_id
    else:
        base_path = Path(lighteval_output_dir) / "details" / model_id
    
    if not base_path.exists():
        raise FileNotFoundError(f"ディレクトリが存在しません: {base_path}")
    
    # 全parquetファイルをスキャン
    all_files = list(base_path.rglob("*.parquet"))
    
    # パターンマッチング
    matched_files = [f for f in all_files if fnmatch(f.name, file_pattern)]
    
    if not matched_files:
        raise FileNotFoundError(
            f"パターンに一致するファイルが見つかりません: {base_path}/**/{file_pattern}"
        )
    
    # ファイル名をタイムスタンプでグループ化
    # ファイル名形式: details_xxx_YYYY-MM-DDTHH-MM-SS.ffffff.parquet
    file_groups = defaultdict(list)
    
    for file_path in matched_files:
        name = file_path.stem  # 拡張子を除く
        parts = name.rsplit('_', 1)
        
        if len(parts) == 2:
            prefix = parts[0] + '_'  # タイムスタンプを除いた部分
            timestamp_str = parts[1]
            file_groups[prefix].append((timestamp_str, file_path))
    
    # 各グループで最新のファイルを選択
    selected_files = []
    for prefix, files in file_groups.items():
        # タイムスタンプで降順ソート（文字列比較で最新を取得）
        files.sort(key=lambda x: x[0], reverse=True)
        latest_file = files[0][1]
        selected_files.append(latest_file)
    
    # 全ファイルを読み込んで結合
    dfs = [pd.read_parquet(f) for f in selected_files]
    return pd.concat(dfs, ignore_index=True)
