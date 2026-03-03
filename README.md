# Hereditary Geometric Meta-RL

Welcome to the codebase of *Hereditary Geometric Meta-RL:
Nonlocal Generalization via Task Symmetries* by Paul Nitschke and Shahriar Talebi, presented at the American Control Conference 2026!

The manuscript can be found 🔗 [here](https://arxiv.org/pdf/2603.00396).

## Setup

To get started, create a virtual environment and install the required dependencies:

```bash
pip install -r requirements.txt
```

## Examples

The folder *examples* contains the 2-D navigation example to get started:

At Meta-Train time:
- *WW_compute_kernel.ipynb*: Learn a kernel distribution from trajectory data.
- *XX_hereditary_geometry_discovery.ipynb*: The main notebook, learns a hereditary geometry from the kernel distributions.

At Meta-Test time:
- *YY_left_action_inference.ipynb*: Infer left actions given trajectory data and a hereditary geometry.
- *ZZ_generate_final_plot.ipynb*: Plots the regret of the hereditary geometry and CCM.

## Questions

Please reach out to 

{first name first author}.{second name first author}@outlook.{german domain}

if you have any questions. We're happy to help.

## File tree

```
hereditary-geometry/
├── requirements.txt
├── README.me
├── constants.py          # Global constants (plotting, dtype)
├── data/                 # Saved outputs (policies, kernels, generators, etc.)
│   └── 2d_navigation/
│       └── circle_task_geometry/
│
├── examples/
│   └── two_d_navigation_task_geo_circle/
│       ├── train.py
│       ├── experiment_argparser.py   # CLI args for experiments
│       ├── oracles.py                # Oracle kernels/charts for 2D nav circle task
│       ├── WW_compute_kernel.ipynb
│       ├── XX_hereditary_geometry_discovery.ipynb
│       ├── YY_left_action_inference.ipynb
│       ├── ZZ_generate_final_plot.ipynb
│       ├── CL_point_env_2/
│       └── pearl_point_env/
│
├── garage/               
│   ├── envs/          
│   ├── torch/Q-functions
│   ├── replay_buffer/
│   ├── sampler/
│   └── experiment/
│
├── src/     
│   ├── learning/
│   │   ├── aa_policy_training/   
│   │   ├── bb_symmetry/         
│   │   ├── cc_hereditary_geometry/  
│   │   ├── inference/           
│   │   ├── main.py             
│   │   └── default_argparser.py  
│   └── plotting/
│       ├── rl/
│       ├── differential/
│       └── loss_over_time.py
│
└── wandb/ 
```