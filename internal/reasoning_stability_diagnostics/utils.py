from pathlib import Path
from typing import Optional
from collections import defaultdict, Counter
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
    # model_id がローカルモデルで絶対パスで書かれている場合は相対パスに変換
    # base_path の構築の際に model_id が絶対パスだと lighteval_output_dir の部分が無視されてしまうため
    model_path = Path(model_id)
    if model_path.is_absolute():
        model_path = model_path.relative_to(model_path.anchor)

    # ベースディレクトリ構築
    if provider:
        base_path = Path(lighteval_output_dir) / "details" / provider / model_path
    else:
        base_path = Path(lighteval_output_dir) / "details" / model_path
    
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


def most_frequent_char_ngram(text: str, n: int = 50) -> dict:
    """
    与えられた text から char-Ngram の最頻値を計算し、
    {"top_ngram": 最頻Ngram, "frequency": 出現回数, "fraction": 最頻Ngramの割合} を返す。

    - text がN文字未満なら {"top_ngram": "", "frequency": 0, "fraction": 0.0} を返す
    - 同率最多が複数ある場合は任意のものを返す
    """
    empty_return = {"top_ngram": "", "frequency": 0, "fraction": 0.0}
    if text is None:
        return empty_return
    
    L = len(text)
    if L < n:
        return empty_return

    counts = Counter()
    for i in range(L - n + 1):
        w = text[i:i+n]
        counts[w] += 1

    # 最頻値（同率の場合は任意）
    top_ngram, top_freq = counts.most_common(1)[0]

    total = L - n + 1  # 全Ngram数
    fraction = top_freq / total
    return {"top_ngram": top_ngram, "frequency": top_freq, "fraction": fraction}
