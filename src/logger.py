from typing import Callable

from ray.tune.logger import UnifiedLogger


def logger_factory(logdir: str) -> Callable:
    results_dir = logdir

    def logger_creator(config: dict) -> UnifiedLogger:
        return UnifiedLogger(config=config, logdir=results_dir)

    return logger_creator
