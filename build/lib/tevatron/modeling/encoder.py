import copy
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import PreTrainedModel, AutoModel
from transformers.file_utils import ModelOutput
from tevatron.arguments import ModelArguments, \
    TevatronTrainingArguments as TrainingArguments

import logging

logger = logging.getLogger(__name__)


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class EncoderPooler(nn.Module):
    def __init__(self, **kwargs):
        super(EncoderPooler, self).__init__()
        self._config = {}
        

    def forward(self, q_reps, p_reps):
        raise NotImplementedError('EncoderPooler is an abstract class')

    def load(self, model_dir: str):
        pooler_path = os.path.join(model_dir, 'pooler.pt')
        if pooler_path is not None:
            if os.path.exists(pooler_path):
                logger.info(f'Loading Pooler from {pooler_path}')
                state_dict = torch.load(pooler_path, map_location='cpu')
                self.load_state_dict(state_dict)
                return
        logger.info("Training Pooler from scratch")
        return

    def save_pooler(self, save_path):
        torch.save(self.state_dict(), os.path.join(save_path, 'pooler.pt'))
        with open(os.path.join(save_path, 'pooler_config.json'), 'w') as f:
            json.dump(self._config, f)


class EncoderModel(nn.Module):
    TRANSFORMER_CLS = AutoModel

    def __init__(self,
                 lm_q: PreTrainedModel,
                 lm_p: PreTrainedModel,
                 pooler: nn.Module = None,
                 untie_encoder: bool = False,
                 negatives_x_device: bool = False,
                 args = None,
                 num_heads = 2, num_layers = 2
                 ):
        super().__init__()
        #print arguments
        # print("lm_q: ", lm_q) # query language model 
        # print("lm_p: ", lm_p) # passage language model
        # both comes from model_name_or_path, which is currently coCondenser-marco-retriever
        # pooler was None
        self.lm_q = lm_q
        self.lm_p = lm_p
        self.original_lm_q =  copy.deepcopy(lm_q)
        self.original_lm_p =  copy.deepcopy(lm_p)
        self.pooler = pooler
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.negatives_x_device = negatives_x_device
        self.untie_encoder = untie_encoder
        if self.negatives_x_device:
            if not dist.is_initialized():
                raise ValueError('Distributed training has not been initialized for representation all gather.')
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

        self.embed_dim = 768 # for coCondenser-marco
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.extend_multi_transformerencoderlayer = torch.nn.TransformerEncoderLayer(self.embed_dim, self.num_heads, batch_first = True, dim_feedforward=3072).to(self.device)
        #if args.identity_bert: # was true in hard-nce-el
        self.extend_multi_transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads, args = args).to(self.device) #TODO: AttributeError!
        self.extend_multi_transformerencoder = torch.nn.TransformerEncoder(self.extend_multi_transformerencoderlayer, self.num_layers).to(self.device)


    def forward(self, query: Dict[str, Tensor] = None, passage: Dict[str, Tensor] = None, pos_passages: Dict[str, Tensor] = None, neg_score: Tensor = None):
        # query's 'input_ids' have shape [128, 32] : seems like batch size is 128 in encoding
        # passage's 'input_ids' have shape [128, 128] : seems like batch size is 128 in encoding

        q_reps = self.encode_query(query) # torch.Size([128, 768]) in encoding, [8, 768] in training
        p_reps = self.encode_passage(passage) # torch.Size([128, 768]) in encoding, [240, 768] in training

        # with open('debugoutput.txt', 'w') as f:
        #     # Redirect the print output to the file for each print statement
        #     print("query: ", file=f)
        #     print(query, file=f)
        #     print("passage: ", file=f)
        #     print(passage, file=f)
        #     print("q_reps size: ", q_reps.size(), file=f)
        #     print("p_reps size: ", p_reps.size(), file=f)
        #     print("q_reps: ", file=f)
        #     print(q_reps, file=f)
        #     print("p_reps: ", file=f)
        #     print(p_reps, file=f)

        # raise Exception("debugging")

        # for inference
        if q_reps is None or p_reps is None: # just for encoding either passage or query, iterating batches
            return EncoderOutput(
                q_reps=q_reps,
                p_reps=p_reps
            )

        # for training, model.train
        if self.training:
            if self.negatives_x_device: # I think this is false for most cases for now
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)
            #print ("negatives_x_deivce", self.negatives_x_device)
            #print ("q_reps", q_reps.shape)
            #print("p_reps", p_reps.shape)

            scores = self.extend_multi(q_reps, p_reps)
            #scores = self.compute_similarity(q_reps, p_reps)
            scores = scores.view(q_reps.size(0), -1)
            # probably ground truth # seems like same index of query and passage is positive
            # since I changed the scores to [8, 30], gold label is zero index.
            target = torch.zeros(scores.size(0), device=scores.device, dtype=torch.long)
            #print("target", target)

            # cross-entropy loss
            student_loss = self.compute_loss(scores, target) 


            # compute teacher scores
            original_q_reps = self.original_encode_query(query) #torch tensor [8, 768] 
            print("org_q_reps size ", original_q_reps.size())
            print("positive passages ", pos_passages) #None이다.. 왜지??
            original_positive_p_reps = self.original_encode_positive_passage(pos_passages) #torch [8, 768]
            print("org_pos_reps size ", original_positive_p_reps.size())
            original_positive_scores = self.compute_positive_similarity(original_q_reps, original_positive_p_reps) # [8,1]
            print("org_pos_scores size ", original_positive_scores.size())
            original_neg_scores = neg_score # list of [8,63] #얘도 None이다..
            print("org_neg_scores size ", original_neg_scores.size())

            #TODO: make teacher_scores of torch tensor [8, 64]. Per query, Add one corresponding positive_score in front of 63 neg_scores
            teacher_scores = torch.cat([original_positive_scores, original_neg_scores], dim = 1) # [8, 64]

            raise Exception("encoder")

            # distillation loss
            teacher_loss = distillation_loss(scores, teacher_scores) 
            
            #TODO: if args, set alpha
            alpha = 0.5
            loss = alpha*student_loss + (1-alpha)*teacher_loss

            print("loss ", loss)
            raise Exception("debugging encoder")
            if self.negatives_x_device:
                loss = loss * self.world_size  # counter average weight reduction
        # for eval, model.eval
        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return EncoderOutput(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    @staticmethod
    def build_pooler(model_args):
        return None

    @staticmethod
    def load_pooler(weights, **config):
        return None

    def encode_passage(self, psg):
        raise NotImplementedError('EncoderModel is an abstract class')

    def encode_query(self, qry):
        raise NotImplementedError('EncoderModel is an abstract class')

    def compute_similarity(self, q_reps, p_reps): # dot product
        return torch.matmul(q_reps, p_reps.transpose(0, 1))

    def compute_positive_similarity(self, q_reps, positive_p_reps): # for KL divergence, teacher score should be also computed by chunks
        num_queries = q_reps.size(0)
        similarities = []

        # only append similarity of ith element of q_reps and ith element of positive_p_reps
        for i in range(num_queries):
            # Compute similarity for each query and its corresponding positive passage
            similarity = torch.matmul(q_reps[i].unsqueeze(0), positive_p_reps[i].unsqueeze(1))
            similarities.append(similarity)

        # Convert list of tensors to a single tensor
        similarities = torch.cat(similarities, dim=0)
        return similarities.squeeze()



    def compute_loss(self, scores, target):
        #print("scores", scores.shape)
        #print("target", target.shape)
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors

    @classmethod
    def build(
            cls,
            model_args: ModelArguments,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # currently, ModelArguments has model_name_or_path from Luyu/co-condenser-marco-retriever, untie_encoder is false
        # load local
        if os.path.isdir(model_args.model_name_or_path):
            if model_args.untie_encoder:
                _qry_model_path = os.path.join(model_args.model_name_or_path, 'query_model')
                _psg_model_path = os.path.join(model_args.model_name_or_path, 'passage_model')
                if not os.path.exists(_qry_model_path):
                    _qry_model_path = model_args.model_name_or_path
                    _psg_model_path = model_args.model_name_or_path
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
            else: #selected 
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        # load pre-trained
        else:
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_args.model_name_or_path, **hf_kwargs)
            lm_p = copy.deepcopy(lm_q) if model_args.untie_encoder else lm_q

        if model_args.add_pooler:
            pooler = cls.build_pooler(model_args)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    @classmethod
    def load(
            cls,
            model_name_or_path,
            model_args,
            train_args: TrainingArguments,
            **hf_kwargs,
    ):
        # load local
        untie_encoder = True
        if os.path.isdir(model_name_or_path): # will be selected if we train or fine-tune
            # TODO: probably will use this, if we make new cme model!
            #raise Exception("debug in model load, case 1")
            _qry_model_path = os.path.join(model_name_or_path, 'query_model')
            _psg_model_path = os.path.join(model_name_or_path, 'passage_model')
            if os.path.exists(_qry_model_path):
                logger.info(f'found separate weight for query/passage encoders')
                logger.info(f'loading query model weight from {_qry_model_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(
                    _qry_model_path,
                    **hf_kwargs
                )
                logger.info(f'loading passage model weight from {_psg_model_path}')
                lm_p = cls.TRANSFORMER_CLS.from_pretrained(
                    _psg_model_path,
                    **hf_kwargs
                )
                untie_encoder = False
            else:
                logger.info(f'try loading tied weight')
                logger.info(f'loading model weight from {model_name_or_path}')
                lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
                lm_p = lm_q
        else: # selected currently, since we use Luyu/co-condenser-marco-retriever
            #raise Exception("debug in model load, case 2")
            logger.info(f'try loading tied weight')
            logger.info(f'loading model weight from {model_name_or_path}')
            lm_q = cls.TRANSFORMER_CLS.from_pretrained(model_name_or_path, **hf_kwargs)
            lm_p = lm_q

        pooler_weights = os.path.join(model_name_or_path, 'pooler.pt')
        pooler_config = os.path.join(model_name_or_path, 'pooler_config.json')
        if os.path.exists(pooler_weights) and os.path.exists(pooler_config):
            logger.info(f'found pooler weight and configuration')
            with open(pooler_config) as f:
                pooler_config_dict = json.load(f)
            pooler = cls.load_pooler(model_name_or_path, **pooler_config_dict)
        else:
            pooler = None

        model = cls(
            lm_q=lm_q,
            lm_p=lm_p,
            pooler=pooler,
            negatives_x_device=train_args.negatives_x_device,
            untie_encoder=model_args.untie_encoder
        )
        return model

    def save(self, output_dir: str):
        if self.untie_encoder:
            os.makedirs(os.path.join(output_dir, 'query_model'))
            os.makedirs(os.path.join(output_dir, 'passage_model'))
            self.lm_q.save_pretrained(os.path.join(output_dir, 'query_model'))
            self.lm_p.save_pretrained(os.path.join(output_dir, 'passage_model'))
        else:
            self.lm_q.save_pretrained(output_dir)
        if self.pooler:
            self.pooler.save_pooler(output_dir)

    def extend_multi(self, q_reps, p_reps):
        # q_reps의 i번째 query 벡터에 대해서, 모든 passage 벡터와 함께 transformer에 넣어
        # 각각의 output을 구한다.
        # 그리고 이 output을 dot product를 통해 similarity score로 바꾼다.
        batch_size = q_reps.size(0)

        # make xs as tensor of q_reps of size [8, 1, 768], when q_reps has size of [8, 768]
        xs = q_reps.unsqueeze(dim=1)
        # make ys as tensor of p_reps of size [8, 30, 768], when p_reps has size of [240, 768], so that one query vector is concatenated with corresponding 30 passage vectors
        ys = p_reps.view(batch_size, -1, self.embed_dim)

        input = torch.cat([xs, ys], dim = 1) # concatenate mention and entity embeddings, should be size [8, 31, 64]
        # Take concatenated tensor as the input for transformer encoder
        attention_result = self.extend_multi_transformerencoder(input)
        # Get score from dot product
        scores = torch.bmm(attention_result[:,0,:].unsqueeze(1), attention_result[:,1:,:].transpose(2,1))
        scores = scores.squeeze(-2) # size [8, 30]

        return scores



        

class IdentityInitializedTransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_head, dim_feedforward=None, dropout=0.1, activation="relu", args = None):
        super(IdentityInitializedTransformerEncoderLayer, self).__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )  
        self.args = args
        # TODO: get real args!!!            
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model, n_head, batch_first = True)
        #if self.args.fixed_initial_weight:
        #    self.weight = torch.tensor([0.]).to(self.device)
        #else:
        self.weight = nn.Parameter(torch.tensor([-5.], requires_grad = True))
            
    def forward(self,src, src_mask=None, src_key_padding_mask=None, is_causal = False):
        out1 = self.encoder_layer(src, src_mask, src_key_padding_mask) # For Lower Torch Version
        # out1 = self.encoder_layer(src, src_mask, src_key_padding_mask, is_causal) # For Higher Torch Version
        sigmoid_weight = torch.sigmoid(self.weight).to(self.device)
        out = sigmoid_weight*out1 + (1-sigmoid_weight)*src
        return out




# Distillation loss function 
def distillation_loss(student_outputs, teacher_outputs, temperature = 1):
    ## Inputs
    # student_outputs (torch.tensor): score distribution from the model trained
    # teacher_outputs (torch.tensor): score distribution from the teacher model
    # labels (torch.tensor): gold for given instance
    # alpha (float): weight b/w two loss terms -> moved to forward function
    # temperature (flost): softmac temperature
    ## Outputs
    # teacher_loss (torch.tensor): KL-divergence b/w student and teacher model's score distribution

    # Teacher loss is obtained by KL divergence
    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if teacher_outputs is not None:
        #teacher_outputs = teacher_outputs[:,:student_outputs.size(1)].to(device
        teacher_loss = nn.KLDivLoss(reduction='batchmean')(nn.functional.log_softmax(student_outputs/temperature, dim=1),
                              nn.functional.softmax(teacher_outputs/temperature, dim=1))
    else: 
        teacher_loss = torch.tensor([0.]).to(device)
    return teacher_loss
