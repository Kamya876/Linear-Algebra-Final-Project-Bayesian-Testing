# Dataset placement

Download the **Cookie Cats A/B testing** CSV from Kaggle (the dataset referenced by the instructor) and save it as
`data/cookie_cats.csv`. The file is ignored by Git (`data/*.csv` in `.gitignore`) — do not commit it.

Columns:
- `userid`: player identifier (unique per row)
- `version`: experiment group, `gate_30` (A) or `gate_40` (B)
- `sum_gamerounds`: rounds played in the first 14 days (a post-assignment variable, not used for the retention analysis)
- `retention_1`: returned 1 day after install (True/False)
- `retention_7`: returned 7 days after install (True/False)

The scripts validate the required columns (`userid, version, retention_1, retention_7`).
