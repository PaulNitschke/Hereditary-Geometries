import argparse

def get_argparser():
    parser = argparse.ArgumentParser(description="Hereditary Geometry Discovery Experiment Configuration")

    # Parameters to create task distribution and train policies.
    parser.add_argument("--train_policies", type=str2bool, default="false", help="Whether to train policies and store them.")
    parser.add_argument("--n_steps_train_pis", type=int, default=100_000, help="Number of steps to train each policy for.")
    parser.add_argument("--n_tasks", type=int, default=4, help="Number of tasks to generate.")
    parser.add_argument("--n_envs", type=int, default=2, help="Number of parallel environments to use when training policies.")


    # Parameters for learning symmetry.
    parser.add_argument("--compute_kernel", type=str2bool, default="false", help="Whether to compute kernel distribution.")
    parser.add_argument("--kernel_dim", type=int, default=1, help="Dimension of the kernel.")
    parser.add_argument("--kernel_dim_T", type=int, default=4, help="Dimension of the commutant kernel of T.")
    parser.add_argument("--which_data_R", #Which data goes in the kernel estimation for R.
                        nargs='+', 
                        choices=['observations', 'actions', 'next_observations'], 
                        default=['next_observations'],
                        help="Input for the kernel learning of R, e.g. p. must be in ['observations', 'actions', 'next_observations'].")
    parser.add_argument("--which_data_T",  #Which data goes in the commutator estimation for T.
                        nargs='+', 
                        choices=['observations', 'actions', 'next_observations'],
                        default=['observations', 'actions'],
                        help="Input for the kernel learning of T, e.g. p. must be in ['observations', 'actions', 'next_observations'].")
    
    parser.add_argument("--epsilon_ball", type=float, default=0.05, help="Tolerance under which f(p') \approx f(p).")
    parser.add_argument("--epsilon_level_set", type=float, default=0.005, help="Ball in which we Taylor approximate f(p).")
    parser.add_argument("--n_neighbors_in_level_set", type=int, default=5, help="Minimum number of samples in the local level set to compute a basis.")
    
    parser.add_argument("--ambient_dim", type=int, default=2, help="Ambient dimension on which the Lie group acts.")
    parser.add_argument("--n_epochs_neural_kernel", type=int, default=2_000, help="Number of epochs to train neural kernel for.")
    parser.add_argument("--lasso_coef_kernel", type=float, default=0.1, help="Lasso coefficient for the neural kernel training.")
    parser.add_argument("--batch_size_neural_kernel", type=int, default=1024, help="Batch size for the neural kernel computation.")


    ######## From here on all parameters for hereditary symmetry discovery.#########
    parser.add_argument("--learn_hereditary_symmetry", type=str2bool, default="true", help="Whether to perform hereditary symmetry discovery.")
    # Main parameters.
    parser.add_argument("--update_chart_every_n_steps", type=int, default=100, help="Update chart every n steps")
    parser.add_argument("--eval_span_how", type=str, choices=["weights", "ortho_comp"], default="weights", help="How to evaluate span")
    parser.add_argument("--n_steps_lgs", type=int, default=20_000, help="Number of optimization steps for the left action discovery (usually higher).")
    parser.add_argument("--n_steps_gen", type=int, default=10_000, help="Number of optimization steps for learning the generator (usually smaller).")
    parser.add_argument("--n_steps_sym", type=int, default=20_000, help="Number of optimization steps for the symmetry discovery (usually medium).")
    parser.add_argument("--log_lg_inits_how", type=str, choices=["log_linreg", "random"], default="log_linreg", help="Initialization method of the log-left actions.")
    parser.add_argument("--sample_data_how", type=str, choices=["uniform_replay", "uniform_manifold", "globally"], default="uniform_manifold", help="Sampling mode for the symmetry discovery.")
    parser.add_argument("--temperature", type=float, default=0.75, help="Temperature for the uniform_manifold data sampling scheme.") #need a very high temperature here as the replay buffer is so one sided.
    parser.add_argument("--neighbors_uniform_sampling", type=int, default=20, help="Number of neighbors to use for the knn estimation used in uniform samplings.")
    
    # General optimization parameters: Batch size, learning rates, lasso coefficients.
    parser.add_argument("--enc_geo_net_sizes", type=list, default=[2,2], help="Network architecture for the encoder of the geometry chart. Same architecture for the decoder.")
    parser.add_argument("--enc_sym_net_sizes", type=int, nargs='+', default=[2,2], help="Network architecture for the encoder of the symmetry chart. Same architecture for the decoder.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--lr_lgs", type=float, default=0.00035, help="Learning rate for the left actions.")
    parser.add_argument("--lr_gen", type=float, default=0.0035, help="Learning rate for the generator loss.")
    parser.add_argument("--lr_chart", type=float, default=0.00085, help="Learning rate for the chart.")
    parser.add_argument("--lasso_coef_lgs", type=float, default=0.5, help="Lasso coefficient for the log-left actions.")
    parser.add_argument("--lasso_coef_generator", type=float, default=0.005, help="Lasso coefficient for the generator.")
    parser.add_argument("--lasso_coef_encoder_decoder", type=float, default=0.0001, help="Lasso coefficient for the encoder and decoder.")
    parser.add_argument("--n_epochs_pretrain_log_lgs", type=int, default=2_500, help="Number of epochs to initialize the log left actions for.")
    parser.add_argument("--n_epochs_init_neural_nets", type=int, default=10_000, help="Number of epochs to initialize encoder and decoder to identity.")

    # Util parameters.
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log_wandb", type=str2bool, default="true", help="Whether to log results to Weights & Biases.")
    parser.add_argument("--log_wandb_gradients", type=str2bool, default="false", help="Whether to log network gradients to Weights & Biases.")
    parser.add_argument("--save_every", type=int, default=10_000, help="Checkpoint frequency.")

    # Debugging parameters.
    parser.add_argument("--use_oracle_kernel", type=str2bool, default="false", help="Whether to use the hard-coded oracle rotation kernel, only used for debugging.")
    parser.add_argument("--use_oracle_T_fns", type=str2bool, default="true", help="Whether to use the hard-coded transition functions, only used for debugging.")
    parser.add_argument("--use_oracle_generator", type=str2bool, default="false", help="Whether to use the hard-coded oracle generator, only used for debugging.")
    parser.add_argument("--use_oracle_sym_chart", type=str2bool, default="false", help="Whether to use the hard-coded oracle symmetry chart, only used for debugging.")

    return parser

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')