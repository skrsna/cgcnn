import dgl.function as fn
import torch 
import dgl 
import torch.nn.functional as F
import pickle 
import numpy as np
import time
import os

class CrystalLoader(object):
    def __init__(self,path_to_pkl):
        self.path_to_pkl = path_to_pkl
        self._load()
    def _load(self):
        with open(self.path_to_pkl,'rb') as infile:
            self.graphs = pickle.load(infile)
        self.targets = torch.Tensor(np.array([graph.target for graph in self.graphs]).reshape(-1, 1).astype(np.float32))
        self.details =[{'adsorbate':graph.adsorbate,
         'miller':graph.miller,
         'comp':graph.comp,
          'mpid':graph.mpid} for graph in self.graphs]
    def __getitem__(self, item):
        g, target = self.graphs[item], self.targets[item]
        return g, target
    def __len__(self):
        return len(self.graphs)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def collate_crystal_graphs_for_regression(data):
    graphs, targets = map(list, zip(*data))
    bg = dgl.batch(graphs)
    bg.set_n_initializer(dgl.init.zero_initializer)
    bg.set_e_initializer(dgl.init.zero_initializer)
    labels = torch.stack(targets, dim=0)
    return bg, labels


def save_checkpoint(state, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    torch.save(state, checkpoint_file)

def save_best(state, directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    else:
        best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, best_model_file)
