from torch.utils.data import Dataset
from torch_geometric.data import Data as Data_pyg
from torch_geometric.data import Dataset as Dataset_pyg
import joblib
import random
embedding_dim = 1024
hidden_size = 1024

class MyDataSet_pyg(Dataset_pyg):
    def __init__(self, path, is_train=False):
        self.data_protein = joblib.load(path)
        self.is_train = is_train
        self.multi_chain = multi_chain

    def __len__(self):
        return len(self.data_protein)

    def __getitem__(self, index):
        embedding = self.data_protein[index][0]
        length = self.data_protein[index][1]
        edge_index = self.data_protein[index][2]
        affinity = self.data_protein[index][-1]
        if random.randint(0, 128) == 0 and self.is_train:
            self.exchange_origin_mut(embedding)
            affinity *= -1
        return embedding, length, Data_pyg(x=edge_index.T), affinity
        # return Data_pyg(x=embedding, edge=edge_index.T, y=affinity)

    def exchange_origin_mut(self, embedding):
        embedding[1], embedding[2] = embedding[2], embedding[1].clone()