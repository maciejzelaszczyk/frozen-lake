import argparse

parser = argparse.ArgumentParser(
    description="Config values for the Frozen Lake 8x8 experiment"
)
parser.add_argument(
    "-chdir",
    "--checkpoint_dir",
    default="outcomes/checkpoints/",
    type=str,
    help="Directory to save experiment checkpoints",
)
parser.add_argument(
    "-env", "--env", default="FrozenLake8x8-v1", type=str, help="Environment to use"
)
parser.add_argument(
    "-fwork",
    "--framework",
    default="torch",
    type=str,
    choices=["tfe", "tf2", "torch"],
    help="Deep learning framework to use",
)
parser.add_argument(
    "-loglev",
    "--log_level",
    default="WARN",
    type=str,
    choices=["INFO", "WARN"],
    help="Logging level",
)
parser.add_argument(
    "-resdir",
    "--results_dir",
    default="outcomes/results/",
    type=str,
    help="Directory to save experiment results",
)
parser.add_argument(
    "-ngpus", "--num_gpus", default=0, type=int, help="Number of GPUs to use"
)
parser.add_argument(
    "-niter", "--num_iter", default=1000, type=int, help="Number of iterations to run"
)
parser.add_argument(
    "-vmax", "--v_max", default=1.0, type=float, help="Max of value function"
)
parser.add_argument(
    "-vmin", "--v_min", default=0.0, type=float, help="Min of value function"
)

args = parser.parse_args()
