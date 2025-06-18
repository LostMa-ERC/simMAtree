from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class AbstractStatsClass(ABC):
    """
    Abstract class to define statistics extract from a generator
    """

    @abstractmethod
    def compute_stats(self, data: Union[List, str, np.ndarray]) -> np.ndarray:
        """
        Compute summary statistics from raw data

        Parameters
        ----------
        data : Union[List, str, np.ndarray]
            Raw data to compute statistics from

        Returns
        -------
        np.ndarray
            Computed summary statistics
        """
        pass

    @abstractmethod
    def get_stats_names(self) -> List[str]:
        """
        Get names of the computed statistics

        Returns
        -------
        List[str]
            Names of statistics in order
        """
        pass

    def get_num_stats(self) -> int:
        """
        Get number of computed statistics
        """
        return len(self.get_stats_names())

    def rescaled_stats(self, stats: np.ndarray) -> list:
        """
        Recover interpretable values from statistics

        Parameters
        ----------
        stats : np.ndarray
            statistics

        Returns
        -------
        list
            recovered statistics
        """
        return stats

    def get_rescaled_stats(self, data: Union[List, str, np.ndarray]) -> np.ndarray:
        """
        Get rescaled data in case statistics are scaled for the inference part
        """
        return self.rescaled_stats(self.compute_stats(data))

    def get_rescaled_stats_names(self) -> List[str]:
        """
        Get names of computed statistics
        """
        return self.get_stats_names()

    def print_stats(self, sample: Union[List, str, np.ndarray]) -> None:
        """
        Print the value of the statistics for a given sample.
        """
        if (
            sample is None
            or sample is False
            or (hasattr(sample, "__len__") and len(sample) == 0)
        ):
            return

        stats = self.get_rescaled_stats(sample)
        names = self.get_rescaled_stats_names()

        if len(stats) != len(names):
            raise ValueError(
                "compute_stats and get_stats_names are not giving the same number of stats"
            )

        for s, n in zip(stats, names):
            print(f"{n}: {s}")
