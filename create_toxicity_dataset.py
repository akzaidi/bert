#!/usr/bin/env python3

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main(source, target):
    source = Path(source).expanduser()
    target = Path(target).expanduser()
    target.mkdir(parents=True, exist_ok=True)

    
    toxic_df = pd.read_csv(str(source / "train.csv"))
    toxic_test = pd.read_csv(str(source / "test.csv"))

    train, dev = train_test_split(toxic_df, test_size=.1, random_state=13)

    train.to_csv(target / "train.tsv", "\t", index=False)
    dev.to_csv(target / "dev.tsv", "\t", index=False)
    toxic_test.to_csv(target / "test.tsv", "\t", index=False)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--source", "-s", type=str, 
                        default="/home/alizaidi/data/kaggle/toxicity",
                        help="Source directory for kaggle toxic comments dataset, should be unzipped")
    parser.add_argument("--target", "-t", type=str,
                        default="/home/alizaidi/data/glue_data/toxicity",
                        help="Target directory for glue versioned toxic dataset")
    args = parser.parse_args()
    main(**vars(args))