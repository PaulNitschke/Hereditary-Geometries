from typing import List
import os
import logging
import argparse

import gym
import torch

from src.utils import SingleLayerNet
from src.learning.aa_policy_training.train_policies import train_and_save_pis_and_buffers
from src.learning.bb_symmetry.utils import compute_and_save_kernel
from src.learning.cc_hereditary_geometry.utils import learn_hereditary_symmetry

def run_hereditary_symmetry_discovery(tasks: List[gym.Env],
                                      save_dir_base: str,
                                      oracles: List[torch.nn.Module],
                                      parser: argparse.ArgumentParser):
    args=parser.parse_args()

    ##############################s########################################################################################################
    if args.train_policies:
        logging.info("Training policies and saving replay buffers for each task...")
        task_dirs=train_and_save_pis_and_buffers(tasks=tasks,
                                                save_dir=save_dir_base,
                                                args=args)
        logging.info("Finished training policies and saving replay buffers for each task.")
    else:
        logging.info("Skipping policy training, using pre-trained policies and replay buffers.")
        task_dirs=[f"{save_dir_base}/task_{i}" for i in range(args.n_tasks)]
        if not all(os.path.exists(dir) for dir in task_dirs):
            raise FileNotFoundError(f"Some task directories do not exist: {task_dirs}")
        
    ######################################################################################################################################

    neural_kernels = []
    for dir in task_dirs:

        if args.compute_kernel:
            logging.info("Computing and saving neural kernel...")
            neural_kernel = compute_and_save_kernel(dir=dir,
                                            args=args)
            logging.info("Finished computing neural kernel.")
        
        else:
            logging.info("Skipping kernel computation, using pre-computed kernel frame.")
            states = torch.load(f"{dir}/states.pt")
            ambient_dim_R = states.shape[1] #TODO, make this more general.
            neural_kernel = SingleLayerNet(ambient_dim=ambient_dim_R, kernel_dim=args.kernel_dim)
            neural_kernel.load_state_dict(torch.load(f"{dir}/neural_kernel.pt"))

        neural_kernels.append(neural_kernel)

    ######################################################################################################################################

    if args.learn_hereditary_symmetry:
        logging.info("Starting hereditary geometry discovery...")
        learn_hereditary_symmetry(dirs=task_dirs,
                                parser=parser,
                                oracles=oracles,
                                tasks_kernel_estimators=neural_kernels)
        logging.info("Finished hereditary geometry discovery.")
    else:
        logging.info("Skipping hereditary geometry discovery, using pre-computed charts.")
        #TODO, here we need to load the pre-computed charts.
        # This is not implemented yet, but we assume that the charts are already computed and saved in the task directories.
        # For now, we just log this information.
        logging.info("Assuming pre-computed charts are available in the task directories.")