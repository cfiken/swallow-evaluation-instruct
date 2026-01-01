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


def normalize_multiple_extractions(extracted_list) -> str:
    """抽出結果をdeduplicateおよびstringに変換
    
    Args:
        extracted_list: 抽出結果のリスト
        
    Returns:
        str: ハッシュ化された結果
    """
    if not extracted_list:
        return "EMPTY"
    if len(extracted_list) == 1:
        return str(extracted_list[0])

    normalized = [str(x) for x in extracted_list]
    unique_extractions = sorted(set(normalized))

    if len(unique_extractions) == 1:
        return unique_extractions[0]

    # 複数ある場合は重複を除去したうえでソートして連結した文字列を返す
    combined = "|".join(unique_extractions)
    return combined


def convert_predictions_to_lookup_keys(
    lst_predictions: list,
    dict_answers: dict,
    num_trial: int,
    prioritize_single_extractions: bool = False
) -> list[str]:
    """抽出された予測リストを回答ルックアップキーのリストに変換する
    
    Args:
        lst_predictions: 抽出された予測のリスト
        dict_answers: 回答の正誤を格納した辞書（キー：回答文字列）
        num_trial: 試行回数
        
    Returns:
        list[str]: ルックアップキーのリスト
        
    Raises:
        ValueError: 抽出された予測が既知の回答にマッチできない場合
    """
    # 予測が期待通りの数の場合、2つずつペアにする
    if len(lst_predictions) == num_trial * 2:
        lst_tup_predictions = list(zip(lst_predictions[0::2], lst_predictions[1::2]))
        lst_prediction_lookup_keys = [normalize_multiple_extractions(tup_pred) for tup_pred in lst_tup_predictions]
    else:
        # 予測数が期待と異なる場合、柔軟にマッチングを試みる
        lst_prediction_lookup_keys = []
        i = 0
        while i < len(lst_predictions):
            # configure lookup key candidates
            lst_predictions_candidate = []
            lst_predictions_candidate.append([lst_predictions[i]])
            if i + 1 < len(lst_predictions):
                lst_predictions_candidate.append([lst_predictions[i], lst_predictions[i+1]])
            if not prioritize_single_extractions:
                # default: 2つの予測を優先的に試す
                lst_predictions_candidate = lst_predictions_candidate[::-1]
            
            for _lst_pred in lst_predictions_candidate:
                _lookup_key = normalize_multiple_extractions(_lst_pred)
                if _lookup_key in dict_answers:
                    lst_prediction_lookup_keys.append(_lookup_key)
                    i += len(_lst_pred)
                    break
            else:
                raise ValueError("Extracted prediction could not be matched to any known answers.")
    
    return lst_prediction_lookup_keys
