from typing import List, Union

import numpy as np
from collections import Counter
import networkx as nx

from .base_stats import AbstractStatsClass

    
class TreeTopologyStats(AbstractStatsClass):
    """
    Summary statistics for tree topologies
    """
    def __init__(self,additional_stats: bool = False):
        pass
    def compute_stats(self, stemmata: list) -> np.ndarray:
        """
        Compute stemmatic summary statistics on a population of trees

        Parameters
        ----------
        stemmata : list[nx.DiGraph]
            list of stemmata given as networkx.DiGraph objects
            (oriented from root to leaves)

        Returns
        -------
        np.ndarray
            Array of computed statistics
        """
        nb_stats = self.get_num_stats()
        node_nb_tot = 0 # total number of nodes in the population
        nb_trees = len(stemmata)
        degree_sequence_pop = [] # degree sequence of the nodes in the population
        degree_sequence_root = [] # list of root degrees over population
        nb_leaves = []
        heights = []
        for g in stemmata:
            node_nb_tot += len(g.nodes())
            degree_sequence_pop.extend([d for n,d in g.out_degree()])
            degree_sequence_root.append(g.out_degree(self._root(g)))
            nb_leaves.append(len(self._leaves(g)))
            heights.append(self._height(g))

        degree_dist = Counter(degree_sequence_pop)
        root_degree_dist = Counter(degree_sequence_root)

        stats = []

        stats.append(degree_dist[2] / node_nb_tot)
        stats.append(degree_dist[3] / node_nb_tot)
        stats.append(degree_dist[4] / node_nb_tot)
        stats.append(root_degree_dist[2] / nb_trees)
        stats.append(root_degree_dist[3] / nb_trees)
        stats.append(root_degree_dist[4] / nb_trees)
        stats.append(np.mean(nb_leaves))
        stats.append(np.mean(heights))

        return np.array(stats, dtype=np.float64)

    def get_stats_names(self) -> List[str]:
        """_
        Get names of computed statistics
        """
        names = [
            "Proportion of nodes with degree 2",
            "Proportion of nodes with degree 3",
            "Proportion of nodes with degree 4",
            "Proportion of nodes with root degree 2",
            "Proportion of nodes with root degree 3",
            "Proportion of nodes with root degree 4",
            "Average number of leaves",
            "Average Height",
        ]

        return names
    
    def _leaves(self, graph: nx.DiGraph) -> List:
        """
        Return the terminal leaves of a tree

        Parameters
        ----------
        graph : nx.DiGraph
            Tree (directed acyclic graph)

        Returns
        -------
        List
            List of node labels of leaves
        """
        return [node for node in graph.nodes() if graph.out_degree(node) == 0]

    def _root(self, graph: nx.DiGraph):
        """
        Return the root of a tree

        Parameters
        ----------
        graph : nx.DiGraph
            Tree (directed acyclic graph)

        Returns
        -------
        node object or None
            Label of the root of graph, None if no root found
        """
        for n in graph.nodes():
            if graph.in_degree(n) == 0:
                return n
        return None

    def _depth(self, graph: nx.DiGraph, node) -> int:
        """
        Returns the depth (distance to root) of a node

        Parameters
        ----------
        graph : nx.DiGraph
        node : any
            label of node belonging to graph

        Returns
        -------
        int
            depth of node within graph
        """
        return nx.shortest_path_length(graph, self._root(graph), node)

    def _height(self, graph: nx.DiGraph) -> int:
        """
        Return the largest depth of a node with a tree

        Parameters
        ----------
        graph : nx.DiGraph

        Returns
        -------
        int
            depth of the leave further fromm the root of graph
        """
        depths = [self._depth(graph, n) for n in graph.nodes()]
        return max(depths)