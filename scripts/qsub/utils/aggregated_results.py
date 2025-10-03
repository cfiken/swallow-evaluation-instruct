import json
from pathlib import Path
from typing import Dict, List, Iterable, Union
import csv
# このスクリプトは。
# model_idsに改行区切りで指定したモデルの評価結果を収集し、並び順を保ったままCSVにscripts/qsub/utils/overall_scores.csvとして保存する。
# これによって、スプレッドシートに簡単に複数モデルをコピペできるようになる。


# How to use:
# 1. model_ids に評価結果を集めたいモデルIDを改行区切りで追加する。
# 2. scripts/generation_settings/generation_settings.json　に モデルIDとcustom_settingsなどの情報を登録する。
# 3. python scripts/qsub/utils/aggregated_results.py を実行する。

model_ids = """
tokyotech-llm/example-7b
tokyotech-llm/example-13b
tokyotech-llm/example-70b
google/example-7b
google/example-13b
"""


def get_overall_scores(
    model_ids: Iterable[str],
    base_dir: Union[str, Path] = ".",
    settings_json_path: Union[str, Path] = "scripts/generation_settings/generation_settings.json",
):
    """
    指定した model_id ごとに aggregated_results.json の overall を読み取って返す。
    """
    base = Path(base_dir)
    
    # generation_settings.json をロード
    settings = json.loads((base / settings_json_path).read_text(encoding="utf-8"))

    results: Dict[str, List[float]] = {}
    tasks = []

    for model_id in model_ids:
        # custom_settings の有無を確認
        custom = settings.get(model_id, {}).get("custom_settings")
        
        if not custom:
            #-iter以降を削って再検索
            model_id_short = model_id.split("-iter")[0]
            custom = settings.get(model_id_short, {}).get("custom_settings")
            

        if custom:
            agg_path = base / "results" / "hosted_vllm" / model_id / custom / "aggregated_results.json"
        else:
            agg_path = base / "results" / "hosted_vllm" / model_id / "aggregated_results.json"



        data = json.loads(agg_path.read_text(encoding="utf-8"))

        overall_raw = data["overall"]
        tasks = data["tasks"]
        overall = [float(x) for x in overall_raw.split(",")]
        results[model_id] = overall

    return results, tasks



ids = [line.strip() for line in model_ids.strip().splitlines() if line.strip()]
scores,tasks = get_overall_scores(ids)

with open("scripts/qsub/utils/overall_scores.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    # write tasks
    writer.writerow(["Model ID"] + tasks)
    for mid in ids:
        row = [mid]  + scores[mid]  # モデルID + スコア列
        writer.writerow(row)
    
print("🎉Overall scores have been written to scripts/qsub/utils/overall_scores.csv")