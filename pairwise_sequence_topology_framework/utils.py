import torch
import numpy as np
from math import sqrt
from scipy import stats
import os
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset,DataLoader

class SiameseTestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='selectivity_train',
                 xd=None, xt_A=None, xt_B=None, y=None, smile_graph=None,
                 compound_ids=None, kinase_A_ids=None, kinase_B_ids=None,
                 max_seq_len=108,
                 transform=None, pre_transform=None):

        super(SiameseTestbedDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.max_seq_len = max_seq_len

        if os.path.isfile(self.processed_paths[0]):
            print(f"  Loading pre-processed {dataset}...")
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            if xd is None:
                raise FileNotFoundError(
                    f"{self.processed_paths[0]} not found. "
                    f"Run create_data.py first."
                )
            print(f"  Processing {dataset}...")
            self.process(xd, xt_A, xt_B, y, smile_graph,
                         compound_ids, kinase_A_ids, kinase_B_ids)
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xt_A, xt_B, y, smile_graph,
                compound_ids, kinase_A_ids, kinase_B_ids):
        assert len(xd) == len(xt_A) == len(xt_B) == len(y), "Mismatched lengths"

        data_list = []
        for i in range(len(xd)):
            smiles = xd[i]
            label = y[i]
            c_size, features, edge_index = smile_graph[smiles]

            if len(edge_index) == 0:
                edge_idx = torch.empty((2, 0), dtype=torch.long)
            else:
                edge_idx = torch.LongTensor(np.array(edge_index)).transpose(1, 0)

            GCNData = DATA.Data(
                x=torch.FloatTensor(np.array(features)),
                edge_index=edge_idx,
                y=torch.FloatTensor([label]),
            )
            GCNData.__setattr__('c_size', torch.LongTensor([c_size]))

            # protein sequences (int16 to save memory)
            GCNData.target_A = torch.ShortTensor(xt_A[i].astype(np.int16))
            GCNData.target_B = torch.ShortTensor(xt_B[i].astype(np.int16))

            # IDs for inference output
            GCNData.compound_id = int(compound_ids[i])
            GCNData.kinase_A_id = str(kinase_A_ids[i])
            GCNData.kinase_B_id = str(kinase_B_ids[i])

            data_list.append(GCNData)

            if (i + 1) % 100000 == 0:
                print(f"    {i+1}/{len(xd)} samples processed...")

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        print(f"  Saving {len(data_list)} samples...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci