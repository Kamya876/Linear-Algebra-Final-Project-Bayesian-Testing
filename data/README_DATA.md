# Dataset placement

Download the Cookie Cats A/B testing CSV from the Kaggle dataset referenced by the instructor. Place the CSV here and name it `cookie_cats.csv`.

Expected commonly used columns:
- `userid`: player identifier
- `version`: experiment group (`gate_30` or `gate_40`)
- `sum_gamerounds`: number of game rounds
- `retention_1`: whether the player returned after one day (0/1)
- `retention_7`: whether the player returned after seven days (0/1)

The scripts validate the columns. If your copy has different column names, edit the constants in `src/real_data.py`.
Do not commit a large raw data file to GitHub unless the instructor explicitly asks for it.
