
import networkx as nx
import torch
import numpy as np
from collections import Iterable

class DataLoader:
    def __init__(self):
        self.graph = nx.Graph()
        self.vertex_to_idx = {}
        self.idx_to_vertex = {}
        self.n_vertices = 0


    def add_indexed_edgelist(self, edgelists):
        idx = 0
        for i, e in enumerate(edgelists):
            if e[0] not in self.vertex_to_idx:
                self.vertex_to_idx[e[0]] = idx
                self.idx_to_vertex[idx] = e[0]
                edgelists[i][0] = idx
                idx+=1
            if e[1] not in self.vertex_to_idx:
                self.vertex_to_idx[e[1]] = idx
                self.idx_to_vertex[idx] = e[1]
                edgelists[i][1] = idx
                idx+=1
        self.graph.add_edges_from(edgelists)
        self.n_vertices = len(edgelists)


    def from_adjacency_list(self, url, directed=False):
        with open(url, 'r+') as f:
            edgelists = [x.split()[:2] for x in f.readlines()]
        
        self.add_indexed_edgelist(edgelists)
                

    def from_csv_edge_list(self, url, directed=False, delimiter=','):
        with open(url, 'r+') as f:
            heading = f.readline().replace('\n','').split(delimiter)
            edgelists = [x.replace('\n','').split(delimiter) for x in f.readlines()]
            attr_edges = []
            for e in edgelists:
                attr_edges.append([e[0], e[1],{heading[2+i]: e[2+i]
                        for i in range(len(heading)-2)}])
        
        self.add_indexed_edgelist(attr_edges)

    

class Embedding:
    def __init__(self, embedding):
        load(embedding)

    def load(self, embedding):
        if isinstance(embedding, torch.nn.sparse.Embedding):
            self.embedding = np.zeros((embedding.num_embeddings,
             embedding.embedding_dim))
            self.num_embeddings = embedding.num_embeddings
            self.embedding_dim = embedding.embedding_dim
        elif isinstance(embedding, list) or isinstance(embedding, np.ndarray):
            self.embedding = np.array(embedding)
            self.num_embeddings = len(embedding)
            self.embedding_dim = len(embedding[0])
    
    def normalize(self):
        for i, e in self.embedding: 
            self.embedding[i, :] = e / np.linalg.norm(e)
    
    def concatenate(self):
        pass