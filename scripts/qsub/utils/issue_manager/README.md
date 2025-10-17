# Issue Manager

issue単位でjobを管理し、ログファイルを確認することなくjobの進捗を把握できるようにするutilsである。

## How to Use
### New Issue
新規issueのqsubは以下の手順で行う。

1. `scripts/qsub/utils/issue_manager/issues`ディレクトリ内で`template`をコピーして任意のissueファイルを作成し、評価したいモデル、タスクなどを`qsub_all.sh`と同じ要領で記入する。

2. 以下のコマンドを実行する。

```sh
bash scripts/qsub/utils/issue_manager/qsub_issue.sh {ISSUE_ID}
```


### Check Status
以下のコマンドでjobの進捗を確認できる。

```sh
bash scripts/qsub/utils/issue_manager/check_status.sh {ISSUE_ID}
```

これを実行すると、`scripts/qsub/utils/issue_manager/visualized`ディレクトリに`{ISSUE_ID}.txt`というファイルが作成される。
このファイル中ではタスクごとに以下のようなステータスがラベリングされている。


```
🟤 queue
🟣 transfer
🔵 running
🟢 done   
🟨 timeout
🟥 error  
⬜️ not submitted
```
下の3つは再度jobを投げる必要があるもので、場合に応じてログファイルを確認していただきたい。

### Resubmit Jobs
再度jobを投げる必要があれば、`qsub_issue.sh`をもう一度実行すれば良い。これによりステータスがtimeout, error, not submittedのタスクが全てqsubされる。
```sh
bash scripts/qsub/utils/issue_manager/qsub_issue.sh {ISSUE_ID}
```
2度目以降の`qsub_issue.sh`でqsubされるjobは`scripts/qsub/utils/issue_manager/resub`ディレクトリに生成される`{ISSUE_ID}.csv`に基づくため、もし投げなくて良いタスクがあればこのcsvのうち不要なモデル・タスクの組を削除すれば良い。
なお、このcsvファイルは`check_status.sh`の実行時に生成されるため、jobを再submitする前にこちらを実行しておくことを推奨する。

## Operation Verification
TSUBAME, ABCIでの動作を検証した。localには対応していない。
また、簡単な検証しか行っていないため使いようによってはバグが発見される可能性は高い。
