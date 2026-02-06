import pickle
from abc import abstractmethod
from typing import Dict, List, Union

import networkx as nx
import numpy as np
import pandas as pd

from src.generator.base_generator import BaseGenerator


class GeneralizedStemmaGenerator(BaseGenerator):
    """
    Generalized stemma generator implementing continuous-time birth-death dynamics

    Implements the complete simulation logic for models with parameters:
    - lda: Initial birth rate (manuscript copying rate)
    - mu: Death rate (manuscript destruction rate)
    - decay: Temporal decay factor for birth rate ∈ [0, 1]
    - decim: Decimation rate at crisis time ∈ [0, 1]

    Uses continuous-time exponential event timing for mathematical precision.
    Subclasses can specialize by fixing certain parameters to zero.
    """

    def __init__(
        self,
        n_init: int,
        Nact: int,
        Ninact: int,
        Ncrisis: int,
        max_pop: int,
    ):
        """
        Initialize the generalized stemma generator

        Parameters
        ----------
        n_init : int
            Number of initial manuscripts (archetype count)
        Nact : int
            Duration of active transmission phase
        Ninact : int
            Duration of inactive phase (pure death)
        Ncrisis : int
            Time when decimation crisis occurs
        max_pop : int
            Maximum population size to prevent memory issues
        """
        self.n_init = n_init
        self.Nact = Nact
        self.Ninact = Ninact
        self.Ncrisis = Ncrisis
        self.max_pop = max_pop

    def generate(
        self,
        rng: np.random.Generator,
        params: Union[list, tuple, dict],
        verbose: bool = False,
    ) -> List[nx.DiGraph]:
        """
        Generate a population of stemma trees

        Parameters
        ----------
        rng : np.random.Generator
            Random number generator
        params : Union[list, tuple, dict]
            Model parameters
        verbose : bool
            Print debug information

        Returns
        -------
        List[nx.DiGraph]
            List of stemma trees with node attributes:
            - 'state' : bool (True if manuscript survives)
            - 'birth_time' : float (creation time)
            - 'death_time' : float (destruction time, if applicable)
        """
        # Extract and validate parameters
        model_params = self._extract_params(params)

        if not self.validate_params(params):
            raise ValueError(f"Invalid parameters for {self.__class__.__name__}")

        stemmas = []

        for _ in range(self.n_init):
            # Generate a complete tree
            complete_tree = self._generate_tree(rng, **model_params)

            # Handle BREAK case (population explosion)
            if complete_tree.graph.get("BREAK", False):
                stemma = self._create_break_tree()
                stemmas.append(stemma)
            elif self._witness_nb(complete_tree) >= 1:
                # Clean the tree to create stemma
                stemma = self._generate_stemma(complete_tree)
                stemmas.append(stemma)
            # If no witnesses, don't add to population

        if verbose:
            total_witnesses = sum(self._witness_nb(s) for s in stemmas)
            print(
                f"Generated {len(stemmas)} trees with {total_witnesses} total witnesses"
            )

        return stemmas

    @abstractmethod
    def _extract_params(self, params: Union[list, tuple, dict]) -> Dict[str, float]:
        """
        Extract model parameters according to the specific model

        Must be implemented by subclasses to define which parameters are used
        and how they are extracted from the input.

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Raw parameters in various formats

        Returns
        -------
        Dict[str, float]
            Dictionary with keys: lda, mu, decay, decim
        """
        pass

    def validate_params(self, params: Union[list, tuple, dict]) -> bool:
        """
        Validate that parameters are physically meaningful

        Parameters
        ----------
        params : Union[list, tuple, dict]
            Parameters to validate

        Returns
        -------
        bool
            True if parameters are valid
        """
        try:
            p = self._extract_params(params)

            # Check non-negativity and finiteness
            for key, value in p.items():
                if value < 0 or np.isnan(value) or np.isinf(value):
                    return False

            # Check decay and decim are in [0, 1]
            if not (0 <= p["decay"] <= 1):
                return False
            if not (0 <= p["decim"] <= 1):
                return False

            return True

        except Exception:
            return False

    def _generate_tree(
        self,
        rng: np.random.Generator,
        lda: float,
        mu: float,
        decay: float,
        decim: float,
    ) -> nx.DiGraph:
        """
        Core continuous-time birth-death simulation with decay and decimation
        Generate a tree (arbre réel) according to birth death model.

        Parameters
        ----------
        lda : float
            birth rate of new node per node per iteration
        mu : float
            death rate of nodes per node per per iteration
        decay: float
            decay rate
        decim: float
            decimation rate
        Tact : int
            number of iterations of the active reproduction phase
        Tinact : int
            number of iterations of the pure death phase (lda is set to 0)
        Tcrisis: int
            number of iterations of the time of the crisis

        Returns
        -------
        G : nx.DiGraph()
            networkx graph object of the generated tree with following node attributes:
                'state' : boolean, True if node living at the end of simulation
                'birth_time' : int
                'death_time' : int

        """
        # Initialize tree
        current_id = 0
        tree = nx.DiGraph()
        tree.add_node(current_id)
        living_nodes = {0}

        birth_time = {0: 0.0}
        death_time = {}

        pop = 1
        t = 0.0
        crisis_happened = False

        # Active phase with temporal decay (continuous time)
        while t < self.Nact:
            lda_1 = (lda * (1 - decay)) + (
                (2 * lda / self.Nact) * (self.Nact - t) * decay
            )
            prob_event = lda_1 + mu
            prob_birth = lda_1 / prob_event if prob_event > 0 else 0

            # Check for extinction
            if pop == 0:
                break

            # Check for population explosion
            if pop > self.max_pop:
                return self._create_break_tree()

            # Time to next event (exponential distribution)
            next_event = rng.exponential(scale=1.0 / (prob_event * pop))

            # Check if event occurs within active phase
            if t + next_event > self.Nact:
                t = self.Nact
                break

            t += next_event

            # Choose random manuscript from living population
            current_node = rng.choice(list(living_nodes))

            # Birth or death event
            if rng.random() < prob_birth:
                # Birth: create new manuscript (copying)
                current_id += 1
                tree.add_node(current_id)
                tree.add_edge(current_node, current_id)
                living_nodes.add(current_id)
                pop += 1
                birth_time[current_id] = t
            else:
                # Death: manuscript is destroyed
                living_nodes.remove(current_node)
                pop -= 1
                death_time[current_node] = t

            # Decimation crisis (punctual event)
            if t > self.Ncrisis and not crisis_happened:
                n_decimated = int(decim * pop)
                if n_decimated > 0 and len(living_nodes) > 0:
                    # Randomly select manuscripts to be destroyed
                    decimated_nodes = rng.choice(
                        list(living_nodes),
                        size=min(n_decimated, len(living_nodes)),
                        replace=False,
                    )
                    for node in decimated_nodes:
                        living_nodes.discard(node)
                        death_time[node] = t
                        pop -= 1
                crisis_happened = True

        # Inactive phase (pure death, continuous time)
        while t < self.Nact + self.Ninact:
            if pop == 0:
                break

            # Time to next death event
            next_event = rng.exponential(scale=1.0 / (mu * pop))

            if t + next_event > self.Nact + self.Ninact:
                t = self.Nact + self.Ninact
                break

            t += next_event
            current_node = rng.choice(list(living_nodes))
            living_nodes.remove(current_node)
            pop -= 1
            death_time[current_node] = t

        # Set node attributes
        living = {n: (n in living_nodes) for n in tree.nodes()}
        nx.set_node_attributes(tree, living, "state")
        nx.set_node_attributes(tree, birth_time, "birth_time")
        nx.set_node_attributes(tree, death_time, "death_time")

        return tree

    def _witness_nb(self, tree: nx.DiGraph) -> int:
        """
        Return the number of living nodes (witnesses) in a tree

        Parameters
        ----------
        tree : nx.DiGraph
            Tree with 'state' node attributes

        Returns
        -------
        int
            Number of surviving manuscripts
        """
        return list(nx.get_node_attributes(tree, "state").values()).count(True)

    def _create_break_tree(self) -> nx.DiGraph:
        """
        Create a special tree indicating population explosion

        Returns
        -------
        nx.DiGraph
            Graph with a special attribute 'BREAK' = True
        """
        tree = nx.DiGraph()
        tree.add_node(0)
        tree.graph["BREAK"] = True
        nx.set_node_attributes(tree, {0: False}, "state")
        nx.set_node_attributes(tree, {0: 0.0}, "birth_time")
        nx.set_node_attributes(tree, {0: 0.0}, "death_time")
        return tree

    def extract_witness_dist(
        self, stemma_population: List[nx.DiGraph]
    ) -> Union[List[int], str]:
        """
        Extract witness distribution from a population of stemmas

        Parameters
        ----------
        stemma_population : List[nx.DiGraph]
            Population of stemmas

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
            living_witnesses = self._witness_nb(stemma)
            if living_witnesses > 0:
                witness_counts.append(living_witnesses)

        return witness_counts

    def _to_DataFrame(self, pop: List[nx.DiGraph]) -> pd.DataFrame:
        """
        Convert population of trees to DataFrame

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: witness_ID, text_ID, birth_time, parent
        """
        text_val = []
        witness_val = []
        birth_val = []
        parent_val = []

        for i, tree in enumerate(pop):
            for node in tree.nodes():
                text_val.append(f"T{i}")
                witness_val.append(f"W{i}-{node}")
                birth_val.append(tree.nodes[node]["birth_time"])
                parents = list(tree.predecessors(node))
                parent = f"W{i}-{parents[0]}" if parents else "ROOT"
                parent_val.append(parent)

        df = pd.DataFrame(
            {
                "witness_ID": witness_val,
                "text_ID": text_val,
                "birth_time": birth_val,
                "parent": parent_val,
            }
        )

        return df

    def save_simul(
        self, pop: List[nx.DiGraph], data_path: str, save_format: str = "csv"
    ) -> pd.DataFrame:
        """
        Save the simulation

        Parameters
        ----------
        pop : List[nx.DiGraph]
            Output of generate function
        data_path : str
            Path to save the file
        save_format : str
            'csv' or 'serialized' (pickle)

        Returns
        -------
        pd.DataFrame
            DataFrame that is saved in the CSV file
        """
        if not pop or pop is None:
            print("An empty population is given so nothing is saved.")
            return None

        if save_format == "csv":
            df = self._to_DataFrame(pop)
            df.to_csv(data_path, sep=";", index=False)
            return df
        elif save_format == "serialized":
            with open(data_path, "wb") as f:
                pickle.dump(pop, f)
            return None
        else:
            raise ValueError(f"Unsupported save format: {save_format}")

    def load_data(self, csv_file: str, csv_sep: str = ";") -> List[nx.DiGraph]:
        """
        Load data from a csv file

        Parameters
        ----------
        csv_file : str
            Path to data
        csv_sep : str
            CSV separator

        Returns
        -------
        List[nx.DiGraph]
            Loaded trees
        """
        df = pd.read_csv(csv_file, sep=csv_sep, engine="python")

        # Group by text_ID to reconstruct trees
        trees = []
        for text_id in df["text_ID"].unique():
            text_df = df[df["text_ID"] == text_id]
            tree = nx.DiGraph()

            # Add nodes
            for _, row in text_df.iterrows():
                node_id = int(row["witness_ID"].split("-")[1])
                tree.add_node(
                    node_id,
                    birth_time=row["birth_time"],
                    state=True,  # Assume all loaded witnesses are alive
                )

                # Add edge if not root
                if row["parent"] != "ROOT":
                    parent_id = int(row["parent"].split("-")[1])
                    tree.add_edge(parent_id, node_id)

            trees.append(tree)

        return trees
