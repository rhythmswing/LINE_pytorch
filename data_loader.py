
import networkx as nx

class DataLoader:
    def __init__(self):
        self.graph = nx.Graph()
        self.vertex_to_idx = {}
        self.idx_to_vertex = {}

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
    def __init__(self):