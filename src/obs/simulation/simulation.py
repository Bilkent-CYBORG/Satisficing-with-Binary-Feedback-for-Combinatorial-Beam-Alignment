"""Base simulation class for running bandit experiments."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

from obs.environment.environment import Environment
from obs.simulation.reward import RewardFunction


class Simulation(ABC):
    """Base class for bandit simulations.

    Orchestrates the interaction between Environment, User, Algorithm, and RewardFunction.
    """

    def __init__(self, environment: Environment, reward_function: RewardFunction):
        """Initialize simulation.

        Parameters
        ----------
        environment : Environment
            Environment instance.
        reward_function : RewardFunction
            RewardFunction instance.
        """
        self.environment = environment
        self.reward_function = reward_function

    @abstractmethod
    def run(self, algorithms: Dict[str, Any], T: int, **kwargs) -> Dict[str, Any]:
        """Run experiment comparing algorithms.

        Parameters
        ----------
        algorithms : dict
            Dict of {"name": Algorithm}.
        T : int
            Number of rounds.
        **kwargs
            Additional parameters.

        Returns
        -------
        dict
            Results dictionary with histories per algorithm.
        """
        pass

    @abstractmethod
    def run_multiple(
        self,
        algorithm_factory: Callable[[], Dict[str, Any]],
        n_experiments: int,
        T: int,
        **kwargs,
    ) -> Dict[str, Any]:
        """Run multiple experiments and aggregate results.

        Parameters
        ----------
        algorithm_factory : callable
            Callable that returns a fresh dict of algorithms.
        n_experiments : int
            Number of experiments.
        T : int
            Number of rounds per experiment.
        **kwargs
            Additional parameters for run().

        Returns
        -------
        dict
            Aggregated results with mean/std statistics.
        """
        pass
