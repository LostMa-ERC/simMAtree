from typing import List, Union

import networkx as nx
import numpy as np

from .base_generator import BaseGenerator


class BirthDeathTree(BaseGenerator):
    """
    Complete tree generator based on a Birth-Death model

    Preserves all tree information (structure, timestamps, states)
    via a directed NetworkX graph.
    """

    def __init__(self, n_init: int, Nact: int, Ninact: int, max_pop: int):
        """
        Initialize the Birth-Death generator for complete trees

        Parameters
        ----------
        n_init : int
            Number of BirthDeath trees to simulate
        Nact : int
            Number of active iterations (birth+death)
        Ninact : int
            Number of inactive iterations (death only)
        max_pop : int
            Maximal population size (to avoid memory issue)
        """
        self.n_init = n_init
        self.Nact = Nact
        self.Ninact = Ninact
        self.max_pop = max_pop

    def generate(
        self, rng: np.random.Generator, params: Union[list, tuple, dict]
    ) -> nx.DiGraph:
        """
        Generate a complete tree according to the Birth-Death model

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        params : Union[list, tuple, dict]
            Model parameters. Must contain at minimum lambda and mu

        Returns
        -------
        nx.DiGraph
            Directed graph representing the tree with attributes:
            - 'state' : bool, True if the node survives to the end
            - 'birth_time' : int, birth time
            - 'death_time' : int, death time (if applicable)
        """
        # Extract parameters
        lda, mu = self._extract_params(params)

        # Validate parameters
        if not self.validate_params(params):
            raise ValueError("Invalid parameters for Birth-Death model")

        stemmas = []

        for i in range(self.n_init):
            # Generate a complete tree
            complete_tree = self._generate_tree(rng, lda, mu)

            # Handle BREAK case
            if complete_tree.graph.get("BREAK", False):
                stemma = self._create_break_tree()

            else:
                stemma = self._generate_stemma(complete_tree)

            stemmas.append(stemma)

        return stemmas

    def _extract_params_old(self, params: Union[list, tuple, dict]) -> tuple:
        """
        Extract lambda and mu from parameters
        """
        if isinstance(params, dict):
            lda = params.get("lda", 0)
            mu = params.get("mu", 0)
        elif isinstance(params, (list, tuple)):
            # Assume order is [LDA, lda, gamma, mu] or [lda, mu]
            if len(params) >= 4:
                lda, mu = params[1], params[3]  # Format [LDA, lda, gamma, mu]
            elif len(params) >= 2:
                lda, mu = params[0], params[1]  # Format [lda, mu]
            else:
                raise ValueError("Not enough parameters provided")
        else:
            raise ValueError(f"Unsupported parameter type: {type(params)}")

        return lda, mu

    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate Birth-Death parameters

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if lambda >= 0, mu >= 0 and lambda > mu
        """
        try:
            lda, mu = self._extract_params(params)
            if lda < 0 or mu < 0 or np.isnan(lda) or np.isnan(mu):
                return False
            # Respect of the constraint
            return lda > mu
        except (ValueError, KeyError, IndexError):
            return False

    def _generate_tree(
        self, rng: np.random.Generator, lda: float, mu: float
    ) -> nx.DiGraph:
        """
        Generate the tree according to the Birth-Death algorithm

        Adapted from the provided generate_tree() function
        """
        current_id = 0
        G = nx.DiGraph()
        G.add_node(current_id)
        living = {0: True}

        birth_time = {0: 0}
        death_time = {}

        pop = 1
        prob_birth = lda
        prob_death = mu

        # Active phase (birth and death)
        for t in range(self.Nact):
            # Create a copy of the node list to avoid modifications during iteration
            current_nodes = list(G.nodes())

            for current_node in current_nodes:
                if not living[current_node]:
                    continue

                r = rng.random()

                # Birth event
                if r < prob_birth:
                    current_id += 1
                    G.add_node(current_id)
                    G.add_edge(current_node, current_id)
                    living[current_id] = True
                    pop += 1
                    birth_time[current_id] = t

                    # Check population limit
                    if pop > self.max_pop:
                        return self._create_break_tree()

                # Death event
                elif prob_birth <= r < (prob_birth + prob_death):
                    living[current_node] = False
                    pop -= 1
                    death_time[current_node] = t

            # Stop if no living population
            if pop == 0:
                break

        # Inactive phase (death only)
        for t in range(self.Ninact):
            current_nodes = list(G.nodes())

            for current_node in current_nodes:
                if not living[current_node]:
                    continue

                r = rng.random()
                if r < prob_death:
                    living[current_node] = False
                    pop -= 1
                    death_time[current_node] = t + self.Nact

            if pop == 0:
                break

        # Assign attributes to nodes
        nx.set_node_attributes(G, living, "state")
        nx.set_node_attributes(G, birth_time, "birth_time")
        nx.set_node_attributes(G, death_time, "death_time")

        return G

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

    def _generate_stemma(self, complete_tree: nx.DiGraph) -> nx.DiGraph:
        """
        Generate stemma from a complete tree

        Parameters
        ----------
        complete_tree : nx.DiGraph
            Complete tree with 'state' node attributes

        Returns
        -------
        nx.DiGraph
            Stemma obtained from complete_tree
        """
        G = nx.DiGraph(complete_tree)
        living = {n: G.nodes[n]["state"] for n in list(G.nodes())}

        # Recursively remove dead leaves until all terminal nodes are living witnesses
        terminal_dead_nodes = [n for n in self._leaves(G) if not living[n]]
        while terminal_dead_nodes:
            for n in terminal_dead_nodes:
                G.remove_node(n)
            # Update living dict after node removal
            living = {n: living[n] for n in G.nodes() if n in living}
            terminal_dead_nodes = [
                n for n in self._leaves(G) if n in living and not living[n]
            ]

        # Remove non-branching consecutive dead nodes
        unwanted_virtual_nodes = [
            n
            for n in list(G.nodes())
            if n in living
            and not living[n]
            and G.out_degree(n) == 1
            and G.in_degree(n) == 1
        ]

        while unwanted_virtual_nodes:
            for n in unwanted_virtual_nodes:
                predecessors = list(G.predecessors(n))
                successors = list(G.successors(n))
                if predecessors and successors:
                    G.add_edge(predecessors[0], successors[0])
                G.remove_node(n)

            # Update living dict and find new unwanted nodes
            living = {n: living[n] for n in G.nodes() if n in living}
            unwanted_virtual_nodes = [
                n
                for n in list(G.nodes())
                if n in living
                and not living[n]
                and G.out_degree(n) == 1
                and G.in_degree(n) == 1
            ]

        # Handle root node if dead and has only one child
        if G.number_of_nodes() > 0:
            root_node = self._root(G)
            if root_node is not None and root_node in living:
                if not living[root_node] and G.out_degree(root_node) == 1:
                    G.remove_node(root_node)

        return G

    def _create_break_tree(self) -> nx.DiGraph:
        """
        Create a special tree indicating that the population limit has been reached

        Returns
        -------
        nx.DiGraph
            Graph with a special attribute 'BREAK' = True
        """
        G = nx.DiGraph()
        G.add_node(0)
        G.graph["BREAK"] = True
        nx.set_node_attributes(G, {0: False}, "state")
        nx.set_node_attributes(G, {0: 0}, "birth_time")
        nx.set_node_attributes(G, {0: 0}, "death_time")
        return G

    def extract_witness_dist(
        self, stemma_population: List[nx.DiGraph]
    ) -> Union[List[int], str]:
        """
        Extract witness distribution from a population of stemmas

        Parameters
        ----------
        stemma_population : List[nx.DiGraph]
            Population of stemmas generated by this class

        Returns
        -------
        Union[List[int], str]
            List of witness counts per tree, or "BREAK" if limit reached
        """
        # Check for BREAK condition
        if len(stemma_population) == 1 and stemma_population[0].graph.get(
            "BREAK", False
        ):
            return "BREAK"

        witness_counts = []

        for stemma in stemma_population:
            # Count living witnesses (leaves) in this stemma
            living_witnesses = len(
                [n for n in self._leaves(stemma) if stemma.nodes[n].get("state", True)]
            )

            # Only count trees with at least one witness
            if living_witnesses > 0:
                witness_counts.append(living_witnesses)

        return witness_counts
