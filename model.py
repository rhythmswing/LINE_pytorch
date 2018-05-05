
import torch.nn as nn
import torch.nn.functional as F
import torch

class LINE1st_Model(nn.Module):
    def __init__(self, n_vertices, n_dimension):
        super(LINE1st_Model, self).__init__()
        self.embeding = nn.Embedding(n_vertices, n_dimension) 

    def forward(self, pos_v, pos_u, weights):

        emb_v = self.embeding(pos_v)
        emb_u = self.embeding(pos_u)
        score = torch.mul(emb_u, emb_v)
        score = torch.sum(score, dim=1)


        score = -F.logsigmoid(score)
        score = torch.mul(score, weights).squeeze()
        score = torch.sum(score)
        return score


class LINE2nd_Model(nn.Module):
    def __init__(self, n_vertices, n_dimension):
        super(LINE2nd_Model, self).__init__()
        self.embedding = nn.Embedding(n_vertices, n_dimension)
        self.embedding_context = nn.Embedding(n_vertices, n_dimension)

    def forward(self, pos_v, pos_u, neg_u, weights):

        emb_v = self.embedding(pos_v)
        emb_u = self.embedding_context(pos_u)
        emb_neg_u = self.embedding_context(neg_u)

        neg_score = torch.bmm(emb_neg_u, emb_v.unsqueeze(2)).squeeze(2)
        neg_score = F.logsigmoid(-neg_score)
        neg_score = torch.sum(neg_score, 1)

        pos_score = torch.mul(emb_v, emb_u).sum(1)

        score = F.logsigmoid(pos_score) + neg_score
        score = -torch.sum(torch.mul(weights, score))

        return score