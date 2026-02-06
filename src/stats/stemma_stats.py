from collections import Counter
from typing import List, Union

import networkx as nx
import numpy as np

from src.stats.base_stats import AbstractStatsClass
from src.utils.stemma_utils import generate_stemma, root


class StemmaStats(AbstractStatsClass):
    """
    Summary statistics for individual stemma trees
    """

    def __init__(self):
        """Initialize stemma statistics calculator"""
        pass

    def compute_stats(self, tree: Union[nx.DiGraph, str, None]) -> np.ndarray:
        """
        Compute summary statistics from a single stemma tree

        Parameters
        ----------
        tree : Union[nx.DiGraph, str, None]
            Stemma tree with 'state' and 'birth_time' node attributes
            Can also be "BREAK" for overflow or None for invalid

        Returns
        -------
        np.ndarray
            Vector of 9 summary statistics, or array of -2 for invalid trees
        """
        # Handle special cases
        if tree is None or tree == "BREAK":
            return np.full(9, -2.0)

        # Count surviving witnesses
        n_living = sum(
            1 for node in tree.nodes() if tree.nodes[node].get("state", False)
        )

        # No survivors - invalid tree
        if n_living == 0:
            return np.full(9, -2.0)

        # Single witness - minimal stats
        if n_living == 1:
            return np.array([1.0, 0.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0])

        # Two witnesses - include timespan
        if n_living == 2:
            birth_times = [
                tree.nodes[n]["birth_time"]
                for n in tree.nodes()
                if tree.nodes[n].get("state", False)
            ]
            timelapse = int(max(birth_times) - min(birth_times))
            return np.array(
                [2.0, float(timelapse), -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]
            )

        # Three or more witnesses - full analysis
        if n_living >= 3:
            return self._compute_full_stats(tree, n_living)

        return np.full(9, -2.0)

    def _compute_full_stats(self, tree: nx.DiGraph, n_living: int) -> np.ndarray:
        """
        Compute full topological statistics for trees with 3+ witnesses

        Parameters
        ----------
        tree : nx.DiGraph
            Complete stemma tree
        n_living : int
            Number of surviving witnesses

        Returns
        -------
        np.ndarray
            Full vector of 9 statistics
        """
        # Generate clean stemma (remove dead leaves and non-branching dead nodes)
        stemma = generate_stemma(tree)
        archetype = root(stemma)

        # Initialize statistics
        birth_times = []
        degrees = []
        direct_filiation_nb = 0
        arch_dists = []

        # Analyze each node
        for node in stemma.nodes():
            degrees.append(stemma.out_degree(node))

            # Count direct filiations (both parent and child are witnesses)
            if node != archetype:
                parents = list(stemma.predecessors(node))
                if parents:
                    father = parents[0]
                    if stemma.nodes[node].get("state", False) and stemma.nodes[
                        father
                    ].get("state", False):
                        direct_filiation_nb += 1

            # Collect birth times of witnesses
            if stemma.nodes[node].get("state", False):
                birth_times.append(stemma.nodes[node]["birth_time"])

            # Calculate distance from archetype
            try:
                path_length = len(
                    nx.shortest_path(stemma, source=archetype, target=node)
                )
                arch_dists.append(path_length)
            except nx.NetworkXNoPath:
                arch_dists.append(0)

        # Compute derived statistics
        timelapse = int(max(birth_times) - min(birth_times)) if birth_times else 0
        deg_dist = Counter(degrees)
        deg1 = deg_dist[1]
        deg2 = deg_dist[2]
        deg3 = deg_dist[3]
        deg4 = deg_dist[4]
        depth = max(arch_dists) if arch_dists else 0
        n_nodes = len(list(stemma.nodes()))

        return np.array(
            [
                float(n_living),
                float(timelapse),
                float(n_nodes),
                float(direct_filiation_nb),
                float(deg1),
                float(deg2),
                float(deg3),
                float(deg4),
                float(depth),
            ]
        )

    def get_stats_names(self) -> List[str]:
        """
        Get names of computed statistics

        Returns
        -------
        List[str]
            Names of the 9 statistics
        """
        return [
            "Number of living witnesses",
            "Timelapse (birth time span)",
            "Number of nodes in stemma",
            "Direct filiation count",
            "Degree 1 count",
            "Degree 2 count",
            "Degree 3 count",
            "Degree 4 count",
            "Maximum depth from archetype",
        ]

    def rescaled_stats(self, stats: np.ndarray) -> np.ndarray:
        """
        No rescaling needed for stemma statistics

        Parameters
        ----------
        stats : np.ndarray
            Raw statistics

        Returns
        -------
        np.ndarray
            Same statistics (no transformation)
        """
        return stats

    def get_rescaled_stats_names(self) -> List[str]:
        """
        Get names of rescaled statistics (same as regular names)

        Returns
        -------
        List[str]
            Names of statistics
        """
        return self.get_stats_names()
