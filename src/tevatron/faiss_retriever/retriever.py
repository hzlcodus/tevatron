import numpy as np
import faiss

import logging
from tqdm import tqdm
from tevatron.modeling.encoder import IdentityInitializedTransformerEncoderLayer

logger = logging.getLogger(__name__)

class CMERetriever: #TODO
    def __init__(self,
                 init_reps: np.ndarray,
                 num_heads = 2, num_layers = 2
                 ):

        self.embed_dim = 768 # for coCondenser-marco
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extend_multi_transformerencoderlayer = IdentityInitializedTransformerEncoderLayer(self.embed_dim, self.num_heads, args = args).to(self.device) #TODO: AttributeError!
        self.extend_multi_transformerencoder = torch.nn.TransformerEncoder(self.extend_multi_transformerencoderlayer, self.num_layers).to(self.device)

    def search(self, q_reps: np.ndarray, k: int):
        scores = self.extend_multi(q_reps, self.reps)
        indices = np.argsort(scores, axis=1)[:, ::-1][:, :k]
        scores = np.sort(scores, axis=1)[:, ::-1][:, :k]
        return scores, indices
    
    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False, q_lookup = None, label_dict = None):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices

    def extend_multi(self, q_reps: np.ndarray, p_reps: np.ndarray): # changed to np.ndarray type
        batch_size = q_reps.shape[0]

        # Convert np.ndarrays to PyTorch tensors
        q_reps_tensor = torch.from_numpy(q_reps)
        p_reps_tensor = torch.from_numpy(p_reps)

        # Original tensor operations converted to PyTorch tensor operations
        xs = q_reps_tensor.unsqueeze(dim=1)
        # make ys as tensor of p_reps of size [8, 30, 768], when p_reps has size of [240, 768], so that one query vector is concatenated with corresponding 30 passage vectors
        ys = p_reps_tensor.view(batch_size, -1, 768)

        input_tensor = torch.cat([xs, ys], dim=1) # concatenate mention and entity embeddings

        # Assuming extend_multi_transformerencoder requires PyTorch tensor
        attention_result = self.extend_multi_transformerencoder(input_tensor)

        # Get score from dot product
        scores_tensor = torch.bmm(attention_result[:, 0, :].unsqueeze(1), attention_result[:, 1:, :].transpose(2, 1))
        scores_tensor = scores_tensor.squeeze(-2)

        # Convert the PyTorch tensor back to a numpy array
        scores = scores_tensor.detach().cpu().numpy()
        return scores


class BaseFaissIPRetriever:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False, q_lookup = None, label_dict = None):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissRetriever(BaseFaissIPRetriever):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
