

from model import LINE1st_Model, LINE2nd_Model
from data_loader import DataLoader

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class LINE:
    def __init__(self, dataloader, dim_1st, dim_2nd):
        self.load(dataloader)
        self.model1st = LINE1st_Model(dataloader.n_vertices, dim_1st)
        self.model2nd = LINE2nd_Model(dataloader.n_vertices, dim_2nd)
    
    def load(self, dataloader):
        assert isinstance(dataloader, DataLoader)
        self.data = dataloader
    
    def train(self, learning_rate=0.0025, batch_size=1, num_iter=5):
        optim1 = optim.SGD(self.model1st, lr=learning_rate)
