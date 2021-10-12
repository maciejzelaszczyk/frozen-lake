# Frozen Lake 8x8

This is a short description of the Frozen Lake 8x8 project.

## Local setup

Below you will find a list of tools used for local setup.

**Docker**. We use *docker-compose* to set up containers. Install *Docker* following the instructions on the *Docker* [website](https://docs.docker.com/compose/install/).

**Pipreqs**. Auto-generate the requirements file for the project based on actual imports, not on installed packages. Install with `pip install pipreqs`.

**Requirements from containers**. For IDEs without *Docker* debug capabilities (e.g. *PyCharm CE*), requirements can be satisfied by running `pip install -r requirements.txt`.

**Pre-commit.** Perform pre-commit code reformatting and checks. Install with `pip install pre-commit`. A full list of pre-commit hooks is specified in the `.pre-commit-config.yaml` file. In particular, we use:

- *Mypy* for static type checking.
- *isort* for grouping and sorting imports.
- *Black* for code formatting.
- *flake8* for non-style checks.

**pytest**. Handle tests. Can be installed via `pip install pytest`. Tests are not included in pre-commit hooks and should be run manually.

## Experiments

We use the `FrozenLake8x8-v1` version of the environment as `FrozenLake8x8-v0` is not compatible with `RLLib`.

We work under the assumption that the environment comes as-is and is not subject to modification. In particular, we do not perform any reward shaping in so far as the environment is concerned.

In principle, it would be possible to shape the reward by altering the behavior of the environment, for instance by applying negative rewards to each time step in order to encourage utilizing the shortes path possible. This could be done by registering a custom environment. It would require adding an `env.py` module:

```python
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv

from src.parser import args


class FrozenLakeShapedEnv(FrozenLakeEnv):
    def __init__(self, env_config: dict):
        super().__init__(**env_config)

    def step(self, action):
        state, reward, done, info = super().step(action)
        if reward == args.fail_reward_default:
            reward_shaped = args.fail_reward_shaped
        elif reward == args.success_reward_default:
            reward_shaped = args.success_reward_shaped
        else:
            reward_shaped = args.timestep_reward_shaped
        return state, reward_shaped, done, info


def env_creator(env_config: dict):
    return FrozenLakeShapedEnv(env_config)
```

Additionally, we would then register the custom environment in `main.py` (only altered lines shown):

```python
from ray.tune.registry import register_env

from env import env_creator


register_env(ENV_SHAPED, env_creator)
agent = ppo.DQNTrainer(config=config, env=ENV_SHAPED, logger_creator=logger_creator)
```

Finally, this would also require us to add arguments to `parser.py`, e.g.:

```python
parser.add_argument("-envsh", "--env_shaped", default="FrozenLake8x8-v1-Shaped", type=str, help="Environment to use (reward shaping applied)")
parser.add_argument("-frsh", "--fail_reward_shaped", default=-10.0, type=float, help="Reward on failure (shaped)")
parser.add_argument("-frdef", "--fail_reward_default", default=0.0, type=float, help="Reward on failure (default)")
parser.add_argument("-srsh", "--success_reward_shaped", default=20.0, type=float, help="Reward on success (shaped)")
parser.add_argument("-srdef", "--success_reward_default", default=1.0, type=float, help="Reward on success (default)")
parser.add_argument("-trsh", "--timestep_reward_shaped", default=-1.0, type=float, help="Reward on timestep (shaped)")
```

Values of `v_max` and `v_min` arguments would also have to be adjusted manually.

Having said that, reward shaping would make the results comparision limited to the modified environment. In order to compare our results to published ones, we restrict our case to the unmodified environment. In particular, we use the `env.env.spec.reward_threshold = 0.85` attribute provided by Gym as a stop criterion. The `0.78` threshold sporadically used in literature is not applicable to the `8x8` environment but rather to the `4x4` case.

We have run a number of experiments with the Frozen Lake 8x8 environment. We have tested:
- Algorithms such as: *A3C*, *APEX-DQN*, *DQN*, *IMPALA*, *PPO*.
- Using attention-based networks instead of regular fully-connected ones.
- The *Intrinsic Curiosity Module* to guide the exploration as the reward is sparse.
- Adjusting the hyperparameters of particular algorithms.

You can inspect the results of the training procedure using Tensorboard: `tensorboard --logdir=./outcomes/results/`.

We find that for the particular `FrozenLake8x8-v1` environment, the most promising results are achieved by `DQN` with vanilla fully-connected neural networks. The most important hyperparameters we have been able to identify relate to the admitted range of the value function governed by the `v_max` and `v_min` parameters. Specifically, tailoring them to the environment by setting `v_max = 1.0` and `v_min = 0.0` helps to get `DQN` off the ground. This is somewhat contrary to the scaling guidelines of `RLLib`, which state that high-throughput algorithms, such as `IMPALA` would be beneficial. It is entirely possible that an exhaustive hyperparameter search would change this conclusion.
