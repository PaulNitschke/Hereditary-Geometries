from src.learning.default_argparser import get_argparser

def get_experiment_argparser():
    parser = get_argparser()

    # Experiment specific parameters
    parser.add_argument("--save_dir_base", type=str, default="data/local/experiment/2d_navigation/circle_task_geometry", help="Path to saving directory for all data.")
    parser.add_argument("--wandb_project_name", type=str, default="two_d_navigation_task_geo_circle", help="Project name for wandb.")

    # Kernel args
    parser.set_defaults(
        kernel_in_R=['next_observations'],
        kernel_in_T=['observations', 'actions', 'next_observations']
    )

    return parser