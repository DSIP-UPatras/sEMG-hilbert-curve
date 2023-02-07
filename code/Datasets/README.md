- Add extracted .mat files in a folder named `__zip__`
- Run `create_dataset_db1.py`. It will take the .mat files from `__zip__` and saved them in a different structure inside /Datasets/Ninapro-DB1
- After running the script, the structure relative to `code/` folder should be:
`Datasets/Datasets/Ninapro-DB1/subject-{:02d}/gesture-{:02d}/rep-{:02d}_{:02d}.mat`
