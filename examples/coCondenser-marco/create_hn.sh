#!/bin/bash

#SBATCH --job-name=example                    # Submit a job named "example"
#SBATCH --nodes=1                             # Using 1 node
#SBATCH --gres=gpu:0                          # Using 1 gpu
#SBATCH --mem=100000MB                         # Using 100GB CPU Memory
#SBATCH --cpus-per-task=8                     # Using 4 maximum processor


python build_train_hn.py --tokenizer_name bert-base-uncased --hn_file /data/cme/train.rank.tsv --qrels /data/cme/marco/qrels.train.tsv --queries /data/cme/marco/train.query.txt --collection /data/cme/marco/corpus.tsv --save_to /data/cme/marco/bert/train-hn-withscore
# ln -s /data/cme/marco/bert/train/* /data/cme/marco/bert/train-hn-withscore
# cd -