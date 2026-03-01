"""
2-D Navigation Task, the agent navigates to a base goal location where the reward function is given by the L_2 distance
between the current location and the goal location.
Task space geometry: Rotate base goal position around origin.
Base task symmetry: Rotation around goal location.

Oracle hereditary Symmetry (to be learned):
- Group: Rotation group SO(2).
- Geometry chart: Identity.
- Symmetry chart: Left translation by base goal location.
"""

import logging
import numpy as np

from garage.envs.point_env import PointEnv
from garage.experiment.deterministic import set_seed

from examples.two_d_navigation_task_geo_circle.experiment_argparser import get_experiment_argparser
from examples.two_d_navigation_task_geo_circle import oracles

from src.learning.main import run_hereditary_symmetry_discovery

CHART_TASK_SPACE_GEO=np.eye(2)

# Create the task distribution.
parser= get_experiment_argparser()
args = parser.parse_args()
logging.info("Creating tasks for 2D navigation task with unit circle around the origin symmetry.")
set_seed(args.seed)
CircleRotation=PointEnv()
train_goal_locations=CircleRotation.sample_tasks(args.n_tasks, chart=CHART_TASK_SPACE_GEO)
tasks=[PointEnv().set_task(goal_location) for goal_location in train_goal_locations]
del CircleRotation

# Create oracles: oracle kernel frame, oracle generator, oracle symmetry and geometry charts.
_oracles = oracles.make_2d_navigation_oracles(train_goal_locations, save_dir_base=args.save_dir_base)
del train_goal_locations

# Run the hereditary symmetry discovery.
logging.info("Starting hereditary symmetry discovery for 2D navigation task with unit circle around the origin symmetry.")
run_hereditary_symmetry_discovery(tasks=tasks,
                                   save_dir_base=args.save_dir_base,
                                   oracles=_oracles,
                                   parser=parser)