import os
from datetime import datetime
import torch

os.environ["WANDB_SILENT"] = "true"
import wandb

from .hereditary_geometry_discovery import HereditaryGeometryDiscovery
from ...utils import get_non_default_args

def learn_hereditary_symmetry(dirs,
          parser,
          oracles: dict,
          tasks_kernel_estimators: list[torch.nn.Module]):
    """
    Helper function to learn hereditary symmetry, loads data, sets up wandb, and trains the model.
    Args:
    -dirs: List of directories containing the replay buffers and frame estimators for each task.
    -oracles: dict, containing the keys 'generator', 'encoder_geo', 'decoder_geo', 'encoder_sym', 'decoder_sym', and 'frames'.
                Either oracle value or None.
    """
    args = parser.parse_args()
    all_tasks_data_R = [torch.load(f"{dir}/all_data_R.pt") for dir in dirs]
    all_tasks_data_T = [torch.load(f"{dir}/all_data_T.pt") for dir in dirs]
    tasks_data_R = [torch.load(f"{dir}/data_R.pt") for dir in dirs]
    tasks_data_T = [torch.load(f"{dir}/data_T.pt") for dir in dirs]
    # tasks_ps = [torch.load(f"{dir}/ps.pt") for dir in dirs]
    # all_tasks_ps = [torch.load(f"{dir}/all_ps.pt") for dir in dirs]
     
        
    # 2. Setup wandb.
    non_default_args= get_non_default_args(parser, args)
    _run_name = '_'.join(f"{k}:{v}" for k, v in non_default_args.items()) if non_default_args else "default"
    run_name = _run_name + '_' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir=os.path.join(os.path.dirname(dirs[0]),"wandb",run_name)
    os.makedirs(save_dir)
    os.makedirs(save_dir + "/pretrain")

    if args.log_wandb:
        wandb.init(project=args.wandb_project_name,name=run_name,config=vars(args))


    # 3. Train.
    her_geo_dis=HereditaryGeometryDiscovery(all_tasks_data_R=all_tasks_data_R,
                                            tasks_data_R=tasks_data_R,
                                            all_tasks_data_T=all_tasks_data_T,
                                            tasks_data_T=tasks_data_T,

                                            tasks_kernel_estimators=tasks_kernel_estimators,
                                            tasks_T_fns=None,
                                            enc_geo_net_sizes=args.enc_geo_net_sizes, 
                                            enc_sym_net_sizes=args.enc_sym_net_sizes,

                                            kernel_dim=args.kernel_dim,
                                            update_chart_every_n_steps=args.update_chart_every_n_steps, 
                                            eval_span_how=args.eval_span_how,
                                            log_lg_inits_how=args.log_lg_inits_how,
                                            sample_data_how=args.sample_data_how,
                                            temperature=args.temperature,

                                            batch_size=args.batch_size, 
                                            lr_lgs=args.lr_lgs,
                                            lr_gen=args.lr_gen,
                                            lr_chart=args.lr_chart,
                                            lasso_coef_lgs=args.lasso_coef_lgs, 
                                            lasso_coef_generator=args.lasso_coef_generator, 
                                            lasso_coef_encoder_decoder=args.lasso_coef_encoder_decoder,
                                            n_epochs_pretrain_log_lgs= args.n_epochs_pretrain_log_lgs, 
                                            n_epochs_init_neural_nets= args.n_epochs_init_neural_nets,
                                            n_neighbors_uniform_sampling=args.neighbors_uniform_sampling,

                                            seed=args.seed, 
                                            log_wandb=args.log_wandb, 
                                            log_wandb_gradients=args.log_wandb_gradients, 
                                            save_every=args.save_every,

                                            save_dir=save_dir,

                                            use_oracle_kernel=args.use_oracle_kernel,
                                            use_oracle_T_fns=args.use_oracle_T_fns,
                                            use_oracle_sym_chart=args.use_oracle_sym_chart,
                                            use_oracle_generator=args.use_oracle_generator,

                                            oracle_kernel=oracles["kernel"],
                                            oracle_T_fns=oracles["T_fns"],
                                            oracle_generator=oracles["generator"], 
                                            oracle_encoder_geo=oracles["encoder_geo"], 
                                            oracle_decoder_geo=oracles["decoder_geo"],
                                            oracle_encoder_sym=oracles["encoder_sym"], 
                                            oracle_decoder_sym=oracles["decoder_sym"]
                                            )
    
    her_geo_dis.optimize(n_steps_lgs=args.n_steps_lgs, n_steps_gen=args.n_steps_gen, n_steps_sym=args.n_steps_sym)
    her_geo_dis.save(f"{save_dir}/results.pt")
    wandb.finish()