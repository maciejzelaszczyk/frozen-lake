import datetime
import logging
import os

import gym
import ray
from gym.wrappers.time_limit import TimeLimit
from ray.rllib.agents.dqn import dqn
from ray.tune.registry import register_env

from src.logger import logger_factory
from src.trainer import AgentTrainer
from tests.tparser import args

# tests explicitly do not have to be DRY

logging.basicConfig(level=logging.INFO)

ray.shutdown()
ray.init(ignore_reinit_error=True)


def test_impossible():
    """Takes some time, even though the environment is a smaller instance."""
    ENV = "FrozenLake-v1"
    NUM_ITER = 20
    IMPOSSIBLE_ENV = "FrozenLake-Impossible"
    IMPOSSIBLE_MAP = ["SHFF", "HFFF", "FFFF", "FFFG"]
    REWARD_THRESHOLD = 0.1

    def impossible_env_creator(config: dict) -> TimeLimit:
        env = gym.make(ENV, desc=IMPOSSIBLE_MAP)
        return env

    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    register_env(IMPOSSIBLE_ENV, impossible_env_creator)

    config: dict = dqn.DEFAULT_CONFIG.copy()
    config["framework"] = args.framework
    config["log_level"] = args.log_level
    config["num_gpus"] = args.num_gpus
    config["v_max"] = args.v_max
    config["v_min"] = args.v_min

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    checkpoint_dir = args.checkpoint_dir + f"{timestamp}/"
    results_dir = args.results_dir + f"{timestamp}/"
    os.makedirs(checkpoint_dir)
    os.makedirs(results_dir)

    logger_creator = logger_factory(logdir=results_dir)
    agent = dqn.DQNTrainer(
        config=config, env=IMPOSSIBLE_ENV, logger_creator=logger_creator
    )
    trainer = AgentTrainer(
        agent=agent,
        num_iter=NUM_ITER,
        reward_threshold=REWARD_THRESHOLD,
        checkpoint_dir=checkpoint_dir,
    )

    success, reward_mean = trainer.train()
    assert success is False
    assert reward_mean == 0.0


def test_small():
    """Takes some time, even though the environment is a smaller instance."""
    ENV = "FrozenLake-v1"
    NUM_ITER = 100
    REWARD_THRESHOLD = 0.5

    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    config: dict = dqn.DEFAULT_CONFIG.copy()
    config["framework"] = args.framework
    config["log_level"] = args.log_level
    config["num_gpus"] = args.num_gpus
    config["v_max"] = args.v_max
    config["v_min"] = args.v_min

    timestamp = datetime.datetime.now(datetime.timezone.utc)
    checkpoint_dir = args.checkpoint_dir + f"{timestamp}/"
    results_dir = args.results_dir + f"{timestamp}/"
    os.makedirs(checkpoint_dir)
    os.makedirs(results_dir)

    logger_creator = logger_factory(logdir=results_dir)
    agent = dqn.DQNTrainer(config=config, env=ENV, logger_creator=logger_creator)
    trainer = AgentTrainer(
        agent=agent,
        num_iter=NUM_ITER,
        reward_threshold=REWARD_THRESHOLD,
        checkpoint_dir=checkpoint_dir,
    )

    success, _ = trainer.train()
    assert success is True
