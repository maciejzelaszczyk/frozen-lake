import logging
from abc import ABC, abstractmethod

from ray.rllib.agents.trainer import Trainer

logging.basicConfig(level=logging.INFO)


class AbstractTrainer(ABC):
    @abstractmethod
    def train(self) -> tuple[bool, float]:
        """Method to train an agent."""


class AgentTrainer(AbstractTrainer):
    def __init__(
        self,
        agent: Trainer,
        num_iter: int,
        reward_threshold: float,
        checkpoint_dir: str,
    ) -> None:
        self.agent = agent
        self.num_iter = num_iter
        self.reward_threshold = reward_threshold
        self.checkpoint_dir = checkpoint_dir
        self.success = False

    def train(self) -> tuple[bool, float]:
        """For a more complex problem, it would be beneficial
        to separate concerns (logging, saving, etc.)."""
        for i in range(self.num_iter):
            result = self.agent.train()
            file_name = self.agent.save(self.checkpoint_dir)
            logging.info(
                f"Iter: {i + 1:3d} | reward_min: {result['episode_reward_min']:6.2f} "
                + f"| reward_mean: {result['episode_reward_mean']:6.2f} "
                + f"| reward_max: {result['episode_reward_max']:6.2f} "
                + f"| len_mean {result['episode_len_mean']:6.2f} | saved: {file_name}"
            )
            if result["episode_reward_mean"] > self.reward_threshold:
                self.success = True
                logging.info("Successfully reached reward threshold.")
                break
        else:
            logging.info("Reward threshold not reached.")
        return self.success, result["episode_reward_mean"]
