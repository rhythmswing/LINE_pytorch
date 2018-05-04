
from data_loader import DataLoader
from model import LINE1st_Model
from model import LINE2nd_Model

from torch.autograd import Variable
import torch
import numpy as np

if __name__ == "__main__":
    d = DataLoader()
    d.from_csv_edge_list('edges')

    model = LINE2nd_Model(10,5)

    v_u = np.array([1,2])
    v_v = np.array([2,3])
    v_neg = np.array([[4,5,5,7],[8,9,5,6]])
    weights = np.array([1,0.5])

    v_u = Variable(torch.from_numpy(v_u))
    v_v = Variable(torch.from_numpy(v_v))
    v_neg = Variable(torch.from_numpy(v_neg))
    weights = Variable(torch.from_numpy(weights))

    model.double()
    model.forward(v_u, v_v, v_neg, weights)
