import logging
import os
import pickle
import sys
from contextlib import nullcontext
from unittest import result

import numpy as np
from tqdm import tqdm

import torch

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers import (
    HfArgumentParser,
)
from tevatron.reranker.data import HFRerankDataset, RerankerInferenceDataset, RerankerInferenceCollator
from tevatron.reranker.modeling import RerankerModel

from tevatron.arguments import ModelArguments, DataArguments, \
    TevatronTrainingArguments as TrainingArguments

logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        model_args: ModelArguments
        data_args: DataArguments
        training_args: TrainingArguments

    if training_args.local_rank > 0 or training_args.n_gpu > 1:
        raise NotImplementedError('Multi-GPU encoding is not supported.')

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )

    num_labels = 1
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir
    )

    model = RerankerModel.load(
        model_name_or_path=model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )

    rerank_dataset = HFRerankDataset(tokenizer=tokenizer, data_args=data_args, cache_dir=data_args.data_cache_dir or model_args.cache_dir)
    rerank_dataset = RerankerInferenceDataset(
        rerank_dataset.process(data_args.encode_num_shard, data_args.encode_shard_index),
        tokenizer, max_q_len=data_args.q_max_len, max_p_len=data_args.p_max_len
    )

    rerank_loader = DataLoader(
        rerank_dataset,
        batch_size=training_args.per_device_eval_batch_size,
        collate_fn=RerankerInferenceCollator(
            tokenizer,
            max_length=data_args.q_max_len+data_args.p_max_len,
            padding='max_length'
        ),
        shuffle=False,
        drop_last=False,
        num_workers=training_args.dataloader_num_workers,
    )
    model = model.to(training_args.device)

    model.eval()
    all_results = {}

    total_batches = len(rerank_loader) # 6974598

    for batch_index, (batch_query_ids, batch_text_ids, q_collated, d_collated) in enumerate(tqdm(rerank_loader)):
        # with open('batchoutput.txt', 'w') as f:
        #     # Redirect the print output to the file for each print statement
        #     print(batch_query_ids, file=f)
        #     print(len(batch_query_ids), file=f)
        #     print(batch_text_ids, file=f)
        #     print(len(batch_text_ids), file=f)
        #     print(q_collated, file=f)
        #     print(q_collated['input_ids'].size(), file=f)
        #     print(d_collated, file=f)
        #     print(d_collated['input_ids'].size(), file=f)


        with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
            with torch.no_grad():
                # for k, v in batch.items():
                #     batch[k] = v.to(training_args.device)
                q_collated = {k: v.to(training_args.device) for k, v in q_collated.items()}
                d_collated = {k: v.to(training_args.device) for k, v in d_collated.items()}

                # print(q_collated['input_ids'][0])
                # print(q_collated['input_ids'][999])

                for key in q_collated:
                    q_collated[key] = q_collated[key][0:1]

                if batch_index == total_batches-1:
                    flag = True
                else:
                    flag = False
                
                model_output = model(batch_query_ids[0], batch_text_ids[0], q_collated, d_collated, flag)

                if model_output.scores is None:
                    continue

                scores = model_output.scores.cpu().detach().numpy() # 현재는 (156, 1) numpy ndarray. i번째 query, passage pair의 score가 저장됨 
                passage_ids = model_output.passage_ids
                prev_query_id = model_output.prev_query_id
                # print("scores", scores)
                # print(scores.shape)
                # raise Exception("with")
                # version1)
                # for i in range(len(scores)):
                #     qid = batch_query_ids[i]
                #     for j in range(1000):
                #         docid = batch_text_ids[i*1000+j]
                #         score = scores[i][j] 
                #         if qid not in all_results:
                #             all_results[qid] = []
                #         all_results[qid].append((docid, score))
                # version2)
                # 이제 score를 (1, 1000) numpy ndarray라고 가정
                qid = batch_query_ids[0]
                if prev_query_id not in all_results:
                    all_results[prev_query_id] = []
                for j in range(len(passage_ids)):
                    docid = passage_ids[j]
                    score = scores[0][j] 
                    all_results[prev_query_id].append((docid, score))
                    

    with open(data_args.encoded_save_path, 'a') as f: # w to a
        for qid in all_results:
            results = sorted(all_results[qid], key=lambda x: x[1], reverse=True)
            for docid, score in results:
                f.write(f'{qid}\t{docid}\t{score}\n')

if __name__ == "__main__":
    main()
