#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import math
import pickle
import numpy as np
import torch
from torch_geometric.data import Data
import torch_geometric.utils as gutil

import matplotlib.pyplot as plt

import networkx as nx


def build_graph(features, edges):
    x = torch.FloatTensor(features)
    src, dst, value = tuple(zip(*edges))
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_index = gutil.to_undirected(edge_index, features.shape[0])
    edge_index = gutil.add_self_loops(edge_index, features.shape[0])
    g = Data(x=x, edge_index=edge_index)
    return g


if __name__ == "__main__":
    season_id = '24581'
    features = np.loadtxt(os.path.join('../tmp', season_id, 'graph_features.txt'))
    edges = np.loadtxt(os.path.join('../tmp', season_id, 'graph_edges.txt'))
    graph = build_graph(features, edges)
    network = gutil.to_networkx(graph.edge_index, num_nodes=graph.num_nodes)

    default_vocab_dictionary_path = os.path.join('../tmp', season_id, 'vocab.dict')
    vocab_dictionary = pickle.load(open(default_vocab_dictionary_path, 'rb'))
    revserse_dictionary = dict(zip(vocab_dictionary.values(), vocab_dictionary.keys()))

    for index in range(network.number_of_nodes()):
        network.nodes[index]['word'] = revserse_dictionary[index]

    print(sorted([(d, revserse_dictionary[n]) for n, d in network.degree()], reverse=True))
