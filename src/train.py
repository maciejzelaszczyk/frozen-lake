import datetime
import os

import gym
import ray
from ray.rllib.agents.dqn import dqn

from src.logger import logger_factory
from src.parser import args
from src.trainer import AgentTrainer

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
agent = dqn.DQNTrainer(config=config, env=args.env, logger_creator=logger_creator)
env = gym.make(
    agent.env_creator.keywords["env_descriptor"]
)  # potential for streamlining
reward_threshold = env.env.spec.reward_threshold
trainer = AgentTrainer(
    agent=agent,
    num_iter=args.num_iter,
    reward_threshold=reward_threshold,
    checkpoint_dir=checkpoint_dir,
)

_, __ = trainer.train()
