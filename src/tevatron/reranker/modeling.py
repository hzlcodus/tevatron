import os
from dataclasses import dataclass
from typing import Dict, Optional

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

    #def forward(self, pair: Dict[str, Tensor] = None):
    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None):
        # ranker_logits = self.hf_model(**pair, return_dict=True).logits
        # if self.train_batch_size:
        #     grouped_logits = ranker_logits.view(self.train_batch_size, -1)
        #     loss = self.cross_entropy(grouped_logits, self.target_label)
        #     return RerankerOutput(
        #         loss = loss,
        #         scores = ranker_logits
        #     )

        with open('pair.txt', 'w') as f:
            torch.set_printoptions(threshold=100000)
            # Redirect the print output to the file for each print statement
            print("query: ", file=f)
            print(query, file=f)
            print("passage: ", file=f)
            print(passage, file=f)

        # Use EncoderModel's forward method
        self.encoder_model.eval()
        encoder_output = self.encoder_model.forward(query=query, passage=passage)
        scores = encoder_output.scores

        # Calculate loss if training and 'loss' is available in encoder_output
        loss = None
        if self.training and hasattr(encoder_output, 'loss'):
            loss = encoder_output.loss

        return RerankerOutput(
            loss = loss,
            scores = ranker_logits
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
