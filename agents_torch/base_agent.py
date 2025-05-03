from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    An abstract base class for all agents_transformers in the system.
    Defines a common interface that all agents_transformers must implement.
    """

    def __init__(self, model=None):
        """
        Agent initialization
        """
        self.model = model

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """
        A method that determines the behavior of the agent depending on the input data.

        :param observation: Input data (e.g. text, dialog context).
        :return: Agent action (e.g., text response).
        """
        pass

    @abstractmethod
    def train_step(self, observations: Any, actions: Any):
        """
        Trains an agent in one step of training.

        :param observations: Input data.
        :param actions: Expected actions.
        """
        pass

    def reset(self):
        """
        Resets the agent state, if any (e.g., in dialog systems).
        """
        pass

    def update(self, feedback: Any):
        """
        Updates the state or parameters of the agent based on feedback.

        :param feedback: Feedback (e.g., evaluation of the quality of the response).
        """
        pass

    def save(self, path: str):
        """
        Saves the agent's parameters or model to a file.

        :param path: Path to save the model.
        """
        pass

    def load(self, path: str):
        """
        Loads the agent's parameters or model from a file.

        :param path: Path to the model file.
        """
        pass
