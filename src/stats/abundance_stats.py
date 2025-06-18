from typing import List, Union

import numpy as np

from .base_stats import AbstractStatsClass


class AbundanceStats(AbstractStatsClass):
    """
    Summary statistics for witness abundance distributions
    """

    def __init__(self, additional_stats: bool = True):
        self.additional_stats = additional_stats

    def compute_stats(self, witness_nb: Union[List, str, np.ndarray]) -> np.ndarray:
        """
        Compute abundance statistics from witness counts

        Parameters
        ----------
        witness_nb : Union[List, str, np.ndarray]
            List of witness counts per text, or "BREAK" for overflow

        Returns
        -------
        np.ndarray
            Array of computed statistics
        """
        nb_stats = self.get_num_stats()

        # Handle special cases
        if not witness_nb:
            return np.zeros(nb_stats)
        elif witness_nb == "BREAK":
            return np.ones(nb_stats)

        witness_nb = np.array(witness_nb, dtype=np.float64)
        nb_texts = witness_nb.size
        nb_witnesses = np.sum(witness_nb)

        stats = []

        # Core statistics (always computed)
        stats.append(nb_witnesses.item() / 1e6)  # Total witnesses (scaled)
        stats.append(nb_texts / 1e6)  # Total texts (scaled)
        stats.append(nb_texts / nb_witnesses.item())  # Texts per witness ratio
        stats.append(
            np.max(witness_nb).item() / nb_witnesses.item()
        )  # Max witnesses proportion
        stats.append(
            np.median(witness_nb).item() / np.max(witness_nb).item()
        )  # Median/max ratio
        stats.append(
            np.sum(witness_nb == 1).item() / nb_texts
        )  # Proportion with 1 witness

        # Additional statistics (optional)
        if self.additional_stats:
            stats.extend(self._compute_additional_stats(witness_nb, nb_texts))

        return np.array(stats, dtype=np.float64)

    def _compute_additional_stats(
        self, witness_nb: np.ndarray, nb_texts: int
    ) -> List[float]:
        """
        Compute additional statistics
        """
        additional = []

        # Proportions with 2, 3, 4 witnesses
        for count in [2, 3, 4]:
            additional.append(np.sum(witness_nb == count).item() / nb_texts)

        # Quantile ratios
        max_witnesses = np.max(witness_nb).item()
        additional.append(np.quantile(witness_nb, 0.75).item() / max_witnesses)
        additional.append(np.quantile(witness_nb, 0.85).item() / max_witnesses)

        # Second largest statistics
        if len(witness_nb) > 1:
            second_largest = np.partition(witness_nb, -2)[-2].item()
            nb_witnesses = np.sum(witness_nb).item()
            additional.append(second_largest / nb_witnesses)
            additional.append(second_largest / max_witnesses)
        else:
            additional.extend([0.0, 0.0])

        return additional

    def rescaled_stats(self, stats: np.ndarray) -> List[int]:
        """
        Recover interpretable values from abundance statistics

        Parameters
        ----------
        stats : np.ndarray
            Computed abundance statistics

        Returns
        -------
        List[int]
            [num_witnesses, num_texts, max_witnesses, median_witnesses, texts_with_one]
        """
        if len(stats) < 6:
            raise ValueError(f"Expected at least 6 statistics, got {len(stats)}")

        num_witnesses = int(stats[0] * 1e6)
        num_texts = int(stats[1] * 1e6)
        max_witnesses = int(stats[3] * stats[0] * 1e6)
        median_witnesses = int(stats[4] * stats[3] * stats[0] * 1e6)
        texts_with_one = int(stats[5] * stats[1] * 1e6)

        return [
            num_witnesses,
            num_texts,
            max_witnesses,
            median_witnesses,
            texts_with_one,
        ]

    def get_stats_names(self) -> List[str]:
        """
        Get names of computed statistics
        """
        names = [
            "Total witnesses (scaled)",
            "Total texts (scaled)",
            "Texts per witness ratio",
            "Max witnesses proportion",
            "Median/max ratio",
            "Proportion with 1 witness",
        ]

        if self.additional_stats:
            names.extend(
                [
                    "Proportions with 2 witnesses",
                    "Proportions with 3 witnesses",
                    "Proportions with 4 witnesses",
                    "Quantile 75 ratios",
                    "Quantile 85 ratios",
                    "Second Largest Witness Proportion",
                    "Second Largest Max Ratio",
                ]
            )
        return names

    def get_rescaled_stats_names(self) -> List[str]:
        """
        Get names of computed statistics
        """
        names = [
            "Number of witnesses",
            "Number of texts",
            "Max. number of witnesses per text",
            "Med. number of witnesses per text",
            "Number of text with one witness",
        ]

        return names
