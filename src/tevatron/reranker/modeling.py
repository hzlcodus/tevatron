import os
from dataclasses import dataclass
from typing import Dict, Optional, List

import torch
from torch import nn, Tensor
from transformers import AutoModelForSequenceClassification, PreTrainedModel, AutoModel
from tevatron.modeling import DenseModel, EncoderModel
from transformers.file_utils import ModelOutput

from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None
    prev_query_id: Optional[int] = None
    passage_ids: Optional[List] = None

class RerankerModel(nn.Module):
    TRANSFORMER_CLS = AutoModel
    def __init__(self, hf_model: PreTrainedModel, train_batch_size: int=None):
        super().__init__()
        self.hf_model = hf_model
        self.encoder_model = DenseModel(
            lm_q=hf_model, 
            lm_p=hf_model,  
        )
        self.hf_model.eval()
        self.encoder_model.eval()
        self.train_batch_size = train_batch_size
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        if train_batch_size:
            self.register_buffer(
                'target_label',
                torch.zeros(self.train_batch_size, dtype=torch.long)
            )
        self.prev_q_id = None
        self.p_id_list = []
        self.qry = None
        self.psg = None
        self.count = 0
        #self.calledcount = 0

    #def forward(self, pair: Dict[str, Tensor] = None):
    def forward(self, q_id: int = None, p_id: int = None, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, flag: bool = False):
        # ranker_logits = self.hf_model(**pair, return_dict=True).logits
        # if self.train_batch_size:
        #     grouped_logits = ranker_logits.view(self.train_batch_size, -1)
        #     loss = self.cross_entropy(grouped_logits, self.target_label)
        #     return RerankerOutput(
        #         loss = loss,
        #         scores = ranker_logits
        #     )
        #self.calledcount = self.calledcount+1

        # Use EncoderModel's forward method
        # version3) 여기서 query_id가 달라질 때까지 이월하기?
        if self.prev_q_id is None:
            self.prev_q_id = q_id
            self.p_id_list.append(p_id)
            self.qry = query
            self.psg = passage
            self.count = self.count+1
            return RerankerOutput(
                loss = None,
                scores = None,
                prev_query_id = None,
                passage_ids = None,
            )
        elif self.prev_q_id != q_id:
            # calculate score
            self.encoder_model.eval()

            # with open('pair.txt', 'a') as f:
            #     torch.set_printoptions(threshold=100000)
            #     # Redirect the print output to the file for each print statement
            #     print("previous query: ", file=f)
            #     print(self.prev_q_id, file=f)
            #     print("new query: ", file=f)
            #     print(q_id, file=f)
            encoder_output = self.encoder_model.forward(query=self.qry, passage=self.psg)
            scores = encoder_output.scores # tensor, [156, 1]
            # with open('count_2.txt', 'a') as f:
            #     # print("scores in modeling", scores, file=f)
            #     # print(scores.size(), file=f)
            #     print("count", self.count, file=f)
                

            temq = self.prev_q_id
            self.prev_q_id = q_id
            temp = self.p_id_list
            self.p_id_list = []
            self.p_id_list.append(p_id)
            self.qry = query
            self.psg = passage
            self.count = 1

            loss = None
            if self.training and hasattr(encoder_output, 'loss'):
                loss = encoder_output.loss
            return RerankerOutput(
                loss = loss,
                scores = scores,
                prev_query_id = temq,
                passage_ids = temp
            )
        elif flag: # doesn't have next query, special handle
            self.prev_q_id = q_id
            self.p_id_list.append(p_id)
            self.qry = query
            self.psg = {key: torch.cat((self.psg[key], passage[key]), dim=0) for key in passage}
            self.count = self.count+1
            # with open('calledcount2.txt', 'a') as f:
            #     # print("scores in modeling", scores, file=f)
            #     # print(scores.size(), file=f)
            #     print("calledcount", self.calledcount, file=f)
            #     print("q_id", q_id, file=f)
                

            self.encoder_model.eval()
            encoder_output = self.encoder_model.forward(query=self.qry, passage=self.psg)
            scores = encoder_output.scores

            loss = None
            if self.training and hasattr(encoder_output, 'loss'):
                loss = encoder_output.loss
            return RerankerOutput(
                loss = loss,
                scores = scores,
                prev_query_id = q_id,
                passage_ids = self.p_id_list
            )
        else: # do not calculate score
            self.prev_q_id = q_id
            self.p_id_list.append(p_id)
            self.qry = query
            self.psg = {key: torch.cat((self.psg[key], passage[key]), dim=0) for key in passage} #concat
            self.count = self.count+1
            return RerankerOutput(
                loss = None,
                scores = None,
                prev_query_id = None,
                passage_ids = None,
            )        


    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        if os.path.isdir(model_args.model_name_or_path):
            logger.info(f'loading model weight from local {model_args.model_name_or_path}')
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)

        else:
            logger.info(f'loading model weight from huggingface {model_args.model_name_or_path}')
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
        model = cls(hf_model=hf_model, train_batch_size=train_args.per_device_train_batch_size)
        return model 

    @classmethod
    def load(
            cls,
            model_name_or_path,
            **hf_kwargs,
    ):

        if os.path.isdir(model_name_or_path):
            logger.info(f'loading model weight from local {model_name_or_path}')
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)

        else:
            logger.info(f'loading model weight from huggingface {model_name_or_path}')
            hf_model = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
        model = cls(hf_model=hf_model)
        return model

    def save(self, output_dir: str):
        self.hf_model.save_pretrained(output_dir)
