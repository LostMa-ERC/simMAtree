from typing import Any, Dict

import torch
from torch.distributions import (
    Beta,
    Categorical,
    Dirichlet,
    Distribution,
    Exponential,
    Gamma,
    Independent,
    LogNormal,
    MultivariateNormal,
    Normal,
    Uniform,
)


class PyTorchPrior:
    """
    Factory class to create PyTorch distributions from configuration
    """

    # Mapping of distribution names to classes
    DISTRIBUTIONS = {
        "uniform": Uniform,
        "normal": Normal,
        "multivariate_normal": MultivariateNormal,
        "independent": Independent,
        "beta": Beta,
        "gamma": Gamma,
        "exponential": Exponential,
        "log_normal": LogNormal,
        "categorical": Categorical,
        "dirichlet": Dirichlet,
    }

    @classmethod
    def create_distribution(
        cls, distribution_name: str, config: Dict[str, Any]
    ) -> Distribution:
        """
        Create a PyTorch distribution from configuration

        Parameters
        ----------
        distribution_name : str
            Name of the distribution (case insensitive)
        config : dict
            Configuration parameters for the distribution

        Returns
        -------
        torch.distributions.Distribution
            Configured PyTorch distribution
        """
        dist_name = distribution_name.lower()

        if dist_name not in cls.DISTRIBUTIONS:
            available = ", ".join(cls.DISTRIBUTIONS.keys())
            raise ValueError(
                f"Unknown PyTorch distribution '{distribution_name}'. Available: {available}"
            )

        dist_class = cls.DISTRIBUTIONS[dist_name]

        # Handle Independent distribution specially
        if dist_name == "independent":
            base_config = config[
                "base_distribution"
            ].copy()  # Make a copy to avoid modifying original
            base_dist_name = base_config.pop("distribution")
            base_dist = cls.create_distribution(base_dist_name, base_config)
            reinterpreted_batch_ndims = config.get("reinterpreted_batch_ndims", 1)
            return Independent(base_dist, reinterpreted_batch_ndims)

        # Convert lists to tensors for all other distributions
        converted_config = {}
        for key, value in config.items():
            if isinstance(value, list):
                converted_config[key] = torch.tensor(value, dtype=torch.float32)
            else:
                converted_config[key] = value

        return dist_class(**converted_config)

    @classmethod
    def get_dimension(cls, distribution) -> int:
        """
        Extract the dimension from a PyTorch distribution

        Parameters
        ----------
        distribution : torch.distributions.Distribution
            PyTorch distribution

        Returns
        -------
        int
            Number of dimensions/parameters
        """
        if hasattr(distribution, "event_shape"):
            # For most distributions, event_shape gives us the dimension
            event_shape = distribution.event_shape
            if len(event_shape) == 0:
                return 1  # Scalar distribution
            elif len(event_shape) == 1:
                return event_shape[0]  # Vector distribution
            else:
                # Multi-dimensional tensor, return total size
                return event_shape.numel()

        # Fallback: try to infer from batch_shape or other attributes
        if hasattr(distribution, "batch_shape") and len(distribution.batch_shape) > 0:
            return distribution.batch_shape[0]

        # Default to 1 if we can't determine
        return 1

    @classmethod
    def is_pytorch_distribution(cls, name: str) -> bool:
        """Check if a distribution name corresponds to a PyTorch distribution"""
        return name.lower() in cls.DISTRIBUTIONS
