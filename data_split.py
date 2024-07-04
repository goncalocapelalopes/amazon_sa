import shutil


if __name__ == "__main__":
    with open(
        "/Users/goncalopes/Documents/sntmnt_analysis/data/raw/train.ft.txt", "r"
    ) as f:
        lines = f.readlines()
    train_lines = lines[:50000]
    val_lines = lines[50000:60000]

    with open(
        "/Users/goncalopes/Documents/sntmnt_analysis/data/processed/train.txt", "w+"
    ) as f:
        f.writelines(train_lines)
    with open(
        "/Users/goncalopes/Documents/sntmnt_analysis/data/processed/val.txt", "w+"
    ) as f:
        f.writelines(val_lines)
    shutil.copyfile(
        "/Users/goncalopes/Documents/sntmnt_analysis/data/raw/test.ft.txt",
        "/Users/goncalopes/Documents/sntmnt_analysis/data/processed/test.txt",
    )
