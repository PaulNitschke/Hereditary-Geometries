import os
import logging
from typing import Literal, List, Tuple, Optional
import time
import copy
import warnings
import scipy

from tqdm import tqdm
import numpy as np
import torch
import wandb

from ...utils import DenseNN
from .initialization import identity_init_neural_net, ExponentialLinearRegressor
from ..bb_symmetry.utils import compute_boltzman_sampling_weights


class HereditaryGeometryDiscovery():

    def __init__(self,
                 all_tasks_data_R: List[torch.tensor],
                 tasks_data_R: List[torch.tensor],
                 all_tasks_data_T: List[torch.tensor],
                 tasks_data_T: List[torch.tensor],
                 tasks_kernel_estimators: List[callable],
                 tasks_T_fns: List[callable],
                 enc_geo_net_sizes: list[int],
                 enc_sym_net_sizes: list[int],

                 kernel_dim: int,
                 update_chart_every_n_steps:int,
                 eval_span_how:Literal['weights', 'ortho_comp'],
                 log_lg_inits_how:Literal['log_linreg', 'random'],
                 sample_data_how:Literal['uniform_replay', 'uniform_manifold', 'globally'],
                 temperature:float,

                 batch_size:int,
                 lr_lgs:float,
                 lr_gen:float,
                 lr_chart:float,
                 lasso_coef_lgs: Optional[float],
                 lasso_coef_generator: Optional[float],
                 lasso_coef_encoder_decoder: Optional[float],
                 n_epochs_pretrain_log_lgs: int,
                 n_epochs_init_neural_nets: int,
                 n_neighbors_uniform_sampling: int,

                 seed:int,
                 log_wandb:bool,
                 log_wandb_gradients:bool,
                 save_dir:str,
                 save_every:int,
                 
                 use_oracle_kernel:bool,
                 use_oracle_T_fns:bool,
                 use_oracle_sym_chart:bool,
                 use_oracle_generator:bool,
                 oracle_kernel:list[callable],
                 oracle_T_fns:list[callable],
                 oracle_generator: torch.tensor,
                 oracle_encoder_geo: torch.nn.Module,
                 oracle_decoder_geo: torch.nn.Module,
                 oracle_encoder_sym: torch.nn.Module,
                 oracle_decoder_sym: torch.nn.Module
                ):
        """Hereditary Geometry Discovery.
        This class implements hereditary symmetry discovery.

        Args:
        - tasks_data_R: list of tensors, each of shape (n_samples, n).
        - tasks_kernel_estimators: list of callables, given a batch of samples from the respective task, returns the frame of the kernel at these samples.
        - kernel_dim: dimension of the kernel.
        - encoder: callable, encodes points into latent space.
        - decoder: callable, decodes points from latent space to ambient space.
        - seed: random seed for reproducibility.
        - lg_inits_how: how to initialize left actions, one of ['random', 'mode', 'zeros']. Mode computes the mode of tasks_data_R and then fits a linear regression between the modes.
        - batch_size: number of samples to use for optimization.
        - lasso_coef_lgs: regularization weight for the lasso regularizer on the left actions, if None, no regularization is applied.

        - oracle_generator: tensor of shape (d, n, n), the generator to be used for symmetry discovery, only use for debugging.

        
        Notation:
        - d: kernel dimension.
        - n: ambient dimension.
        - b: batch size.
        - N: number of tasks.
        """

        self.all_tasks_data_R= all_tasks_data_R
        self.tasks_data_R= tasks_data_R
        self.all_tasks_data_T= all_tasks_data_T
        self.tasks_data_T= tasks_data_T
        self.tasks_kernel_estimators=tasks_kernel_estimators
        self.tasks_T_fns=tasks_T_fns

        self.oracle_generator= oracle_generator
        self.encoder_geo=DenseNN(enc_geo_net_sizes)
        self.encoder_sym=DenseNN(enc_sym_net_sizes)
        self.base_task_index=0
        self._log_wandb_every=500
        self._n_epochs_pretrain_log_lgs=n_epochs_pretrain_log_lgs
        self._n_epochs_init_neural_nets=n_epochs_init_neural_nets

        self.kernel_dim=kernel_dim
        self._update_chart_every_n_steps=update_chart_every_n_steps
        self._eval_span_how=eval_span_how
        self._log_lg_inits_how=log_lg_inits_how
        self._sample_data_how=sample_data_how
        self._temperature=temperature
        self._n_neighbors_uniform_sampling=n_neighbors_uniform_sampling

        self.batch_size=batch_size
        self._lr_lgs=lr_lgs
        self._lr_kgs=lr_lgs
        self._lr_gen=lr_gen
        self._lr_chart=lr_chart
        self._lasso_coef_lgs=lasso_coef_lgs
        self._lasso_coef_generator = lasso_coef_generator
        self._lasso_coef_encoder_decoder=lasso_coef_encoder_decoder
        
        self.seed=seed
        self._log_wandb=log_wandb
        self._log_wandb_gradients=log_wandb_gradients
        self._save_every= save_every
        self._save_dir=save_dir
        
        self._use_oracle_kernel=use_oracle_kernel
        self._use_oracle_T_fns=use_oracle_T_fns
        self._use_oracle_sym_chart=use_oracle_sym_chart
        self._use_oracle_generator=use_oracle_generator

        self._oracle_kernel=oracle_kernel
        self._oracle_T_fns=oracle_T_fns
        self._oracle_encoder_sym= oracle_encoder_sym
        self._oracle_decoder_sym= oracle_decoder_sym
        self._oracle_encoder_geo= oracle_encoder_geo
        self._oracle_decoder_geo= oracle_decoder_geo

        self._validate_inputs()

        self._global_step_wandb=0
        self._n_samples_R, self.dim_S = tasks_data_R[0].shape
        self._n_samples_T=tasks_data_T[0].shape[0]
        self.dim_A=tasks_data_T[0].shape[1] - self.dim_S
        self._n_tasks=len(tasks_data_R)
        self.task_idxs = list(range(self._n_tasks))
        self.task_idxs.remove(self.base_task_index)

        if self._use_oracle_kernel:
            logging.warning("Using oracle kernel frames.", stacklevel=0)
            self.kernel_frame_base_task= self._oracle_kernel[self.base_task_index]
            self.kernel_frame_i_tasks= [self._oracle_kernel[i] for i in self.task_idxs]      
        else:    
            self.kernel_frame_base_task= self.tasks_kernel_estimators[self.base_task_index]
            self.kernel_frame_i_tasks= [self.tasks_kernel_estimators[i] for i in self.task_idxs]

        if self._use_oracle_T_fns:
            logging.warning("Using oracle transition functions.", stacklevel=0)
            self.T_fn_base_task = self._oracle_T_fns[self.base_task_index]
            self.T_fn_i_tasks= [self._oracle_T_fns[i] for i in self.task_idxs]  
        else:    
            self.T_fn_base_task = self.tasks_T_fns[self.base_task_index]
            self.T_fn_i_tasks= [self.tasks_T_fns[i] for i in self.task_idxs]


        # Store losses and diagnostics.
        self._losses, self._diagnostics={}, {}
        _losses_names=["left_actions","left_actions_tasks_reg",
                       "generator_span","generator_weights","reconstruction_geo","generator_reg",
                       "symmetry","reconstruction_sym","symmetry_reg", "oracle_symmetry_span"]
        _diagnostics_names= ["cond_num_generator", "frob_norm_generator",
                                "encoder_loss_oracle_geo", "decoder_loss_oracle_geo",
                                "encoder_loss_oracle_sym", "decoder_loss_oracle_sym"]
        self._losses = {name: [0.0] for name in _losses_names}
        self._diagnostics = {name: [0.0] for name in _diagnostics_names}
        self._losses["left_actions_tasks"] = [np.zeros(self._n_tasks-1, dtype=np.float32)]

        torch.manual_seed(seed)

    
    def evaluate_lgs(self,
                             ps: torch.Tensor, 
                             log_lgs: torch.Tensor, 
                             encoder_geo: torch.nn.Module,
                             decoder_geo: torch.nn.Module,
                             track_loss:bool=True) -> float:
        """Computes kernel alignment loss of all left-actions."""
        # 1. Push-forward
        # lgs=torch.linalg.matrix_exp(log_lgs.param)
        lgs = [torch.linalg.matrix_exp(log_lg.param) for log_lg in log_lgs]


        # def encoded_left_action(ps):
        #     #TODO, this is currently only for 1-D lie groups, the computed jacobian is of shape (b,N,m,n).
        #     """Helper function that lets the exponential of the log-left action act on ps, represented in the current chart.
        #     Used to compute Jacobians."""
        #     tilde_ps=encoder_geo(ps)
        #     #batch-dimension of ps is dropped here as vmap processes the tensors one-by-one.
        #     lg_tilde_ps = torch.einsum("Nmn,n->Nm", lgs, tilde_ps)
        #     return decoder_geo(lg_tilde_ps)
        
        def encoded_left_action(lg, ps):
            #TODO, this is currently only for 1-D lie groups, the computed jacobian is of shape (b,N,m,n).
            """Helper function that lets the exponential of the log-left action act on ps, represented in the current chart.
            Used to compute Jacobians."""
            tilde_ps=encoder_geo(ps)
            #batch-dimension of ps is dropped here as vmap processes the tensors one-by-one.
            lg_tilde_ps = torch.einsum("mn,n->Nm", lg, tilde_ps)
            return decoder_geo(lg_tilde_ps)

        tilde_ps=encoder_geo(ps)
        # lg_tilde_ps = torch.einsum("Nmn,bn->Nbm", lgs, tilde_ps)
        lg_tilde_ps = [torch.einsum("mn,bn->bm", lgs[i], tilde_ps) for i in range(self._n_tasks-1)]
        # lg_ps = decoder_geo(lg_tilde_ps)
        lg_ps = [decoder_geo(lg_tilde_ps[i]) for i in range(self._n_tasks-1)]

        # 2. Sample tangent vectors and push-forward tangent vectors
        frame_ps= self.kernel_frame_base_task(ps)
        frames_i_lg_ps = torch.stack([self.kernel_frame_i_tasks[idx_task](lg_ps[idx_task]) for idx_task in range(self._n_tasks-1)], dim=0)


        # jac_lgs = torch.vmap(torch.func.jacrev(encoded_left_action))(ps)
        jac_lgs = [torch.vmap(torch.func.jacrev(encoded_left_action, argnums=1), in_dims=(None, 0))(lgs[i], ps) for i in range(self._n_tasks-1)]
        # Jac_lgs_frame_ps = torch.einsum("bNmn,bdn->Nbdm", jac_lgs, frame_ps)
        jac_lgs_frame_ps = [torch.einsum("bmn,bdn->bdm", jac_lgs[i], frame_ps) for i in range(self._n_tasks-1)]
        # self.task_losses = -torch.abs(torch.cosine_similarity(Jac_lgs_frame_ps, frames_i_lg_ps, dim=-1)).mean(-1).mean(-1)
        self.task_losses = -torch.stack([torch.abs(torch.cosine_similarity(jac_lgs_frame_ps[i], frames_i_lg_ps[i], dim=-1)).mean(-1).mean(-1) for i in range(self._n_tasks-1)], dim=0)


        # # 3. Compute orthogonal complement and projection loss.
        # # _, ortho_frame_i_lg_ps = self._project_onto_vector_subspace(frames_i_lg_ps, Jac_lgs_frame_ps)
        # _, ortho_lgs_frame_ps = self._project_onto_vector_subspace(Jac_lgs_frame_ps, frames_i_lg_ps)
        # mean_ortho_comp = lambda vec: torch.norm(vec, dim=(-1)).mean(-1).mean(-1)
        # self.task_losses = mean_ortho_comp(ortho_lgs_frame_ps)
        self.task_losses_reg = self._lasso_coef_lgs*torch.norm(log_lgs.param, p=1, dim=(-1, -2)).mean(-1)

        loss_reconstruction_geo= torch.linalg.vector_norm(ps - decoder_geo(encoder_geo(ps)), dim=-1).mean()
        if track_loss:
            self._losses["left_actions_tasks"].append(self.task_losses.detach().cpu().numpy())
            self._losses["left_actions_tasks_reg"].append(self.task_losses_reg.detach().cpu().numpy())
            self._losses["left_actions"].append(self.task_losses.mean().detach().cpu().numpy())
            self._losses["reconstruction_geo"].append(loss_reconstruction_geo.detach().cpu().numpy())

        # Compare the current encoder to an oracle encoder, only used for debugging.
        if self._oracle_encoder_geo is not None and self._oracle_decoder_geo is not None:
            with torch.no_grad():
                encoder_loss_oracle_geo = torch.norm(self._oracle_encoder_geo(ps) - encoder_geo(ps), dim=-1).sum(0)
                decoder_loss_oracle_geo = torch.norm(self._oracle_decoder_geo(ps) - decoder_geo(ps), dim=-1).sum(0)
            if track_loss:
                self._diagnostics["encoder_loss_oracle_geo"].append(encoder_loss_oracle_geo.detach().cpu().numpy())
                self._diagnostics["decoder_loss_oracle_geo"].append(decoder_loss_oracle_geo.detach().cpu().numpy())

        return self.task_losses.mean() + loss_reconstruction_geo + self.task_losses_reg
    

    def evaluate_T(self,
                     s_and_as: torch.tensor,
                     log_lgs: torch.tensor,
                     log_kgs: torch.Tensor) -> float:
        """Find left actions Lg and Kg such that the transition functions of all tasks are equal to the T fn of the base task."""
        
        states = s_and_as[:,:self.dim_S]
        actions = s_and_as[:,self.dim_S:]

        lgs=torch.linalg.matrix_exp(log_lgs.param)
        kgs=torch.linalg.matrix_exp(log_kgs.param)

        # Model based approach for T where we directly compare the transition functions.
        s_prime_base = self.T_fn_base_task(states, actions)
        s_tilde_tasks = torch.einsum("Nmn,bn->Nbm", lgs, states)
        a_tilde_tasks = torch.einsum("Nmn,bn->Nbm", kgs, actions)
        s_prime_i_tasks_lst = [self.T_fn_i_tasks[idx_task](s_tilde_tasks[idx_task], a_tilde_tasks[idx_task]) for idx_task in range(self._n_tasks-1)]
        s_prime_i_tasks = torch.stack(s_prime_i_tasks_lst, dim=0)
        lg_s_prime_base = torch.einsum("Nmn,bn->Nbm", lgs, s_prime_base)

        task_losses_T = torch.linalg.vector_norm(s_prime_i_tasks-lg_s_prime_base, dim=-1).mean(-1)
        # self.task_losses_kg_reg = self._lasso_coef_lgs*torch.norm(log_kgs.param, p=1, dim=(-1, -2)).mean(-1)
        return task_losses_T.mean(-1)
        # return task_losses_T.mean(-1) + self.task_losses_kg_reg


    def _project_onto_tensor_subspace(self, tensors: torch.tensor, basis:torch.tensor) -> Tuple[torch.tensor, torch.tensor]:
        """
        Projects 2-tensors onto a d-dimensional subspace of 2-tensors.
        Args:
        - tensors: torch.tensor of shape (b, n, n), b two-tensors
        - basis: torch.tensor of shape (d, n, n), a d-dimensional vector space of two-tensors, given by its basis.

        Returns: 
        - proj: torch.tensor of shape (b, n, n), the projection of tensors onto the subspace spanned by basis
        - ortho_comp: torch.tensor of shape (b, n, n), the orthogonal complement of tensors with respect to the subspace spanned by basis.
        """
        b,n,_= tensors.shape
        d,_,_= basis.shape
        tensors_flat=tensors.reshape(b, n*n)
        basis_flat=basis.reshape(d, n*n)


        proj_vecs_flat, ortho_vecs_flat = self._project_onto_vector_subspace(tensors_flat, basis_flat)
        proj = proj_vecs_flat.reshape(b, n, n)
        ortho_comp = ortho_vecs_flat.reshape(b, n, n)

        with torch.no_grad():
            s = torch.linalg.svdvals(basis_flat)
            self._diagnostics["cond_num_generator"].append((s.max()/s.min()).item())
            self._diagnostics["frob_norm_generator"].append(torch.mean(torch.linalg.matrix_norm(basis)).item())

        return proj, ortho_comp
    

    def evaluate_generator_span(self, 
                                log_left_actions: torch.Tensor, 
                                generator: torch.nn.Module, 
                                track_loss:bool=True)->float:
        """
        Evalutes whether all log-left-actions are inside the span of the generator: log_lgs \in span(generator).
        log_lgs are frozen in this loss function (and hence a detached tensor).
        """

        if self._eval_span_how == "weights":
            log_lgs_hat = torch.einsum("Nd,dmn->Nmn", self.weights_lgs_to_gen.param, generator)
            loss_weights = torch.mean((log_lgs_hat - log_left_actions) ** 2)
            loss_reg=self._lasso_coef_generator*torch.sum(torch.abs(generator)) + self._lasso_coef_lgs*torch.sum(torch.abs(self.weights_lgs_to_gen.param))
            loss = loss_weights + loss_reg

            with torch.no_grad():
                _, ortho_log_lgs_generator=self._project_onto_tensor_subspace(log_left_actions, generator)
                loss_span=torch.mean(torch.linalg.matrix_norm(ortho_log_lgs_generator),dim=0)

        elif self._eval_span_how == "ortho_comp":
            _, ortho_log_lgs_generator=self._project_onto_tensor_subspace(log_left_actions, generator)
            loss_span=torch.mean(torch.linalg.matrix_norm(ortho_log_lgs_generator),dim=0)
            loss_reg=self._lasso_coef_generator*torch.sum(torch.abs(generator))
            loss_weights = torch.tensor(0.0)
            loss = loss_span + loss_reg

        if track_loss:
            self._losses["generator_span"].append(loss_span.detach().cpu().numpy())
            self._losses["generator_weights"].append(loss_weights.detach().cpu().numpy())
            self._losses["generator_reg"].append(loss_reg.detach().cpu().numpy())

        return loss
    

    def _l1_penalty(self, model):
        """L1 Penalty on model parameters."""
        return sum(p.abs().sum() for p in model.parameters())
        
        

    def _project_onto_vector_subspace(self, vecs, basis):
        """
        Projects 1-tensors onto a d-dimensional subspace of 1-tensors.
        vecs: tensor of shape (N, b, d, n)
        basis: tensor of shape (N, b, d, n)
        Returns:
        - proj_vecs: tensor of shape (N, b, d, n)
        - ortho_vecs: tensor of shape (N, b, d, n)
        """
        basis_t = basis.transpose(-2, -1)
        G = torch.matmul(basis, basis_t)
        G_inv = torch.linalg.pinv(G)
        P = torch.matmul(basis_t, torch.matmul(G_inv, basis))
        proj_vecs = torch.matmul(vecs, P)
        return proj_vecs, vecs-proj_vecs
    

    
        
    def take_step_R(self):
        """Update the left actions under a frozen chart."""
        ps = self.tasks_data_R[self.base_task_index][torch.randint(0, self._n_samples_R, (self.batch_size,))] #TODO, think about which points to use here.

        for log_lg in self.log_lgs:
            for p in log_lg.parameters(): p.requires_grad = True
        for p in self.encoder_geo.parameters(): p.requires_grad = False
        for p in self.decoder_geo.parameters(): p.requires_grad = False

        # 1. Step left-actions, left-action loss is independent of generator.
        [optim.zero_grad for optim in self.optimizer_lgs]     
        loss_left_action = self.evaluate_lgs(ps=ps, 
                                                      log_lgs=self.log_lgs, 
                                                      encoder_geo=self.encoder_geo, 
                                                      decoder_geo=self.decoder_geo)
        loss_left_action.backward()
        [optim.step() for optim in self.optimizer_lgs]


    def take_step_T(self):
        """Update the left actions of the transition function under a frozen chart."""
        s_and_as = self.tasks_data_T[self.base_task_index][torch.randint(0, self._n_samples_T, (self.batch_size,))] #TODO, think about which points to use here.

        for p in self.log_lgs.parameters(): p.requires_grad = False
        for p in self.log_kgs.parameters(): p.requires_grad = True
        for p in self.encoder_geo.parameters(): p.requires_grad = False
        for p in self.decoder_geo.parameters(): p.requires_grad = False

        # 1. Step left-actions, left-action loss is independent of generator.
        self.optimizer_kgs.zero_grad()        
        loss_left_action = self.evaluate_T(s_and_as=s_and_as,
                                            log_lgs=self.log_lgs, 
                                            log_kgs=self.log_kgs)
        loss_left_action.backward()
        self.optimizer_kgs.step()


    def take_step_generator_left_actions(self):
        """Steps the generator given the log left actions and the symmetry charts."""   

        for log_lg in self.log_lgs:
            for p in log_lg.parameters(): p.requires_grad = False
        for p in self.generator_S.parameters(): p.requires_grad = True
        for p in self.generator_A.parameters(): p.requires_grad = True

        self.optimizer_generator_S.zero_grad()
        self.optimizer_generator_A.zero_grad()
        generator_S_normed = self.generator_S.param / torch.linalg.matrix_norm(self.generator_S.param)
        generator_A_normed = self.generator_A.param / torch.linalg.matrix_norm(self.generator_A.param)
        loss_span_S = self.evaluate_generator_span(generator=generator_S_normed, log_left_actions=self.log_lgs.param)
        loss_span_A = self.evaluate_generator_span(generator=generator_A_normed, log_left_actions=self.log_kgs.param)

        (loss_span_S + loss_span_A).backward()

        self.optimizer_generator_S.step()
        self.optimizer_generator_A.step()
        

    # def take_step_chart_geo(self):
    #     """Update the geometry chart under frozen leftactions."""

    #     ps = self.tasks_data_R[self.base_task_index][torch.randint(0, self._n_samples_R, (self.batch_size,))] #TODO, probably need to sample from all tasks for this to globally train encoder/ decoder.

    #     for p in self.log_lgs.parameters(): p.requires_grad = False
    #     for p in self.encoder_geo.parameters(): p.requires_grad = True
    #     for p in self.decoder_geo.parameters(): p.requires_grad = True

    #     self.optim_encoder_geo.zero_grad()
    #     self.optim_decoder_geo.zero_grad()
    #     loss_left_action = self.evaluate_lgs(ps=ps,
    #                                                   log_lgs=self.log_lgs, 
    #                                                   encoder_geo=self.encoder_geo, 
    #                                                   decoder_geo=self.decoder_geo)
    #     loss_left_action.backward()
    #     self.optim_encoder_geo.step()
    #     self.optim_decoder_geo.step()

    
    # def take_step_geometry(self):
    #     """Update the left actions under a frozen chart."""
    #     ps = self.tasks_data_R[self.base_task_index][torch.randint(0, self._n_samples_R, (self.batch_size,))] #TODO, think about which points to use here.
    #     s_and_as = self.tasks_data_T[self.base_task_index][torch.randint(0, self._n_samples_T, (self.batch_size,))] #TODO, think about which points to use here.

    #     for p in self.log_lgs.parameters(): p.requires_grad = True
    #     for p in self.log_kgs.parameters(): p.requires_grad = True
    #     for p in self.generator_S.parameters(): p.requires_grad = True
    #     for p in self.generator_A.parameters(): p.requires_grad = True
    #     for p in self.encoder_geo.parameters(): p.requires_grad = False
    #     for p in self.decoder_geo.parameters(): p.requires_grad = False

    #     # 1. Step left-actions, left-action loss is independent of generator.
    #     self.optimizer_lgs.zero_grad()
    #     # self.optimizer_kgs.zero_grad()
    #     self.optimizer_generator_S.zero_grad()
    #     # self.optimizer_generator_A.zero_grad()
    #     # self.optim_encoder_geo.zero_grad()
    #     # self.optim_decoder_geo.zero_grad()

    #     loss_left_action = self.evaluate_lgs(ps=ps, 
    #                                         log_lgs=self.log_lgs, 
    #                                         encoder_geo=self.encoder_geo, 
    #                                         decoder_geo=self.decoder_geo)
        
    #     # loss_T_fn = self.evaluate_T(s_and_as=s_and_as, log_lgs=self.log_lgs, log_kgs=self.log_kgs)

    #     generator_S_normed = self.generator_S.param / torch.linalg.matrix_norm(self.generator_S.param)
    #     # generator_A_normed = self.generator_A.param / torch.linalg.matrix_norm(self.generator_A.param)
    #     loss_generator_S = self.evaluate_generator_span(generator=generator_S_normed, log_left_actions=self.log_lgs.param)
    #     # loss_generator_A = self.evaluate_generator_span(generator=generator_A_normed, log_left_actions=self.log_kgs.param)

    #     # (loss_left_action + loss_T_fn + loss_generator_S + loss_generator_A).backward()
    #     (loss_left_action + loss_generator_S).backward()
    #     # loss_left_action.backward()
    #     self.optimizer_lgs.step()
    #     # self.optimizer_kgs.step()
    #     self.optimizer_generator_S.step()
    #     # self.optimizer_generator_A.step()
    #     # self.optim_encoder_geo.step()
    #     # self.optim_decoder_geo.step()
        

    def optimize(self, 
                 n_steps_lgs:int,
                 n_steps_gen:int,
                 n_steps_sym:int):
        
        """Main optimization loop."""
        self.progress_bar_lgs = tqdm(range(n_steps_lgs), desc="Learn log lg left actions...")
        self.progress_bar_kgs = tqdm(range(n_steps_lgs), desc="Learn log kg left actions...")
        self.progress_bar_gen = tqdm(range(n_steps_gen), desc="Learn lg generator...")

        # 1. Initialize left-actions, encoder and decoder.
        self._init_optimization()

        # 2. Learn left actions and their chart.
        logging.info("Learning log left actions and chart.")

        #1. Learn left actions.
        for idx in self.progress_bar_lgs:

            self.take_step_R()

            if idx % self._log_wandb_every == 0:
                self._log_to_wandb(step=self._global_step_wandb)
                self._global_step_wandb+=self._log_wandb_every
                time.sleep(0.05)

        for idx in self.progress_bar_kgs:
            self.take_step_T()

            # if idx%self._save_every == 0 and self._save_dir is not None:
            #     os.makedirs(f"{self._save_dir}/geo/step_{idx}") if not os.path.exists(f"{self._save_dir}/geo/step_{idx}") else None
            #     self.save(f"{self._save_dir}/geo/step_{idx}/results.pt")

        #2. Learn generators W_S and W_A that contain L_g and K_g.
        for idx in self.progress_bar_gen:
            self.take_step_generator_left_actions()

        logging.info("Finished learning log left actions and generators..")


    def save(self, path: str):
        """Saves the model to a file."""
        torch.save({
            'lgs': self.lgs,
            'kgs': self.kgs,
            'generator_S': self.generator_S.param,
            'generator_A': self.generator_A.param,
            'log_lgs': self.log_lgs.param,
            'log_kgs': self.log_kgs.param,
            'lgs_inits': self.lgs_inits,
            'encoder_geo_state_dict': self.encoder_geo.state_dict(),
            'decoder_geo_state_dict': self.decoder_geo.state_dict(),
            'encoder_sym_state_dict': self.encoder_sym.state_dict(),
            'decoder_sym_state_dict': self.decoder_sym.state_dict(),
            'losses': self._losses,
            'seed': self.seed
        }, path)


    def _init_log_lgs_linear_reg(self, epochs, log_wandb:bool=False):
        """Fits log-linear regressors to initialize left actions."""
        logging.info("Fitting log-linear regressors to initialize left actions.")
        self._log_lg_inits = [
            ExponentialLinearRegressor(input_dim=self.dim_S, seed=self.seed, log_wandb=log_wandb, task_idx=idx_task).fit(
                X=self.all_tasks_data_R[0], Y=self.all_tasks_data_R[idx_task], epochs=epochs
            )
            for idx_task in self.task_idxs]
        logging.info("Finished fitting log-linear regressors to initialize left actions.")   
        self._global_step_wandb+=len(self.task_idxs)*self._n_epochs_pretrain_log_lgs    
        # return torch.stack(self._log_lg_inits, dim=0)
        return self._log_lg_inits

    @property
    def lgs(self):
        return torch.linalg.matrix_exp(self.log_lgs.param)
    

    @property
    def kgs(self):
        return torch.linalg.matrix_exp(self.log_kgs.param)
    

    @property
    def lgs_inits(self):
        return torch.linalg.matrix_exp(self._log_lg_inits)
    

    @property
    def kgs_inits(self):
        return torch.linalg.matrix_exp(self._log_kg_inits)


    @property
    def losses(self):
        """Returns all losses."""
        return {
            "left_actions": np.array(self._losses["left_actions"][1:]),
            "left_actions_tasks": np.array(self._losses["left_actions_tasks"][1:]),
            "left_actions_tasks_reg": np.array(self._losses["left_actions_tasks_reg"][1:]),
            "generator": np.array(self._losses["generator_span"][1:]),
            "symmetry": np.array(self._losses["symmetry"][1:]),
            "reconstruction_sym": np.array(self._losses["reconstruction_sym"][1:]),
            "symmetry_reg": np.array(self._losses["symmetry_reg"][1:]),
        }


    def _init_optimization(self):
        """Initializes the optimization: initializes the left-actions, encoder and decoder and defines the optimizers."""

        # if self._sample_data_how == "uniform_manifold":
        #     self.uniform_replay_weights = compute_boltzman_sampling_weights(self.tasks_data_R[self.base_task_index], temperature=self._temperature, k=self._n_neighbors_uniform_sampling)

        class TensorToModule(torch.nn.Module):
            def __init__(self, tensor):
                """Converts a tensor to a PyTorch module for easier gradient tracking. Used for the log-left actions and the generator."""
                super().__init__()
                self.param=torch.nn.Parameter(tensor)
                torch.nn.utils.parametrizations.weight_norm(self, name='param', dim=0)

        # 1. Log-left actions.
        if self._log_lg_inits_how == 'log_linreg':
            self._log_lg_inits=self._init_log_lgs_linear_reg(log_wandb=self._log_wandb,epochs=self._n_epochs_pretrain_log_lgs)
            self._log_kg_inits = torch.randn(size=(self._n_tasks-1, self.dim_A, self.dim_A))

        elif self._log_lg_inits_how == 'random':
            self._log_lg_inits = torch.randn(size=(self._n_tasks-1, self.dim_S, self.dim_S))
            self._log_kg_inits = torch.randn(size=(self._n_tasks-1, self.dim_A, self.dim_A))
        # self.log_lgs=TensorToModule(self._log_lg_inits.clone())
        self.log_lgs=[TensorToModule(lg.clone()) for lg in self._log_lg_inits]

        self.log_kgs=TensorToModule(self._log_kg_inits.clone())
        # self.optimizer_lgs = torch.optim.Adam(self.log_lgs.parameters(),lr=self._lr_lgs)
        self.optimizer_lgs = [torch.optim.Adam(lg.parameters(), lr=self._lr_lgs) for lg in self.log_lgs]
        self.optimizer_kgs = torch.optim.Adam(self.log_kgs.parameters(),lr=self._lr_kgs)
        
        # 2. Generator.
        _generator_S=torch.stack([torch.eye(self.dim_S) for _ in range(self.kernel_dim)])
        _generator_A=torch.stack([torch.eye(self.dim_A) for _ in range(self.kernel_dim)])
        _generator_S = self.oracle_generator if self._use_oracle_generator else _generator_S
        _generator_A = self.oracle_generator if self._use_oracle_generator else _generator_A #TODO, this should be a different oracle generator.
        assert _generator_S.shape == (self.kernel_dim, self.dim_S, self.dim_S), "Generator must be of shape (d, n, n)." #TODO, this should rather be called Lie group dimension.
        self.generator_S= TensorToModule(_generator_S)
        self.generator_A= TensorToModule(_generator_A)
        self.optimizer_generator_S=torch.optim.Adam(self.generator_S.parameters(), lr=self._lr_gen)
        self.optimizer_generator_A=torch.optim.Adam(self.generator_A.parameters(), lr=self._lr_gen)

        if self._eval_span_how=="weights":
            self.weights_lgs_to_gen = TensorToModule(torch.randn(size=(self._n_tasks-1, self.kernel_dim), requires_grad=True))
            self.optimizer_weights_lgs_to_gen= torch.optim.Adam(self.weights_lgs_to_gen.parameters(), lr=self._lr_gen)

        # 3. Charts.
        _identity_chart= identity_init_neural_net(self.encoder_geo, tasks_data=self.tasks_data_R, name="chart placeholder", log_wandb=self._log_wandb, 
                                               n_steps=self._n_epochs_init_neural_nets)
        self._global_step_wandb+=self._n_epochs_init_neural_nets

        self.encoder_geo = copy.deepcopy(_identity_chart)
        self.decoder_geo = copy.deepcopy(_identity_chart)
        self.encoder_sym = copy.deepcopy(_identity_chart) if not self._use_oracle_sym_chart else self._oracle_encoder_sym
        self.decoder_sym = copy.deepcopy(_identity_chart) if not self._use_oracle_sym_chart else self._oracle_decoder_sym
        del _identity_chart

        self.optim_encoder_geo = torch.optim.Adam(self.encoder_geo.parameters(), lr=self._lr_chart)
        self.optim_decoder_geo = torch.optim.Adam(self.decoder_geo.parameters(), lr=self._lr_chart)
        self.optim_encoder_sym = torch.optim.Adam(self.encoder_sym.parameters(), lr=self._lr_chart)
        self.optim_decoder_sym = torch.optim.Adam(self.decoder_sym.parameters(), lr=self._lr_chart)


    def _validate_inputs(self):
        """Validates user inputs."""
        assert self._log_lg_inits_how in ['random', 'log_linreg'], "_log_lg_inits_how must be one of ['random', 'log_linreg']."
        assert len(self.tasks_data_R) == len(self.tasks_kernel_estimators), "Number of tasks and frame estimators must match."
        logging.warning("\n\n\nUsing oracle generator\n\n\n") if self._use_oracle_generator else None
        logging.warning("\n\n\nUsing oracle symmetry chart\n\n\n", stacklevel=0) if self._use_oracle_sym_chart else None
        logging.warning("\n\n\nSampling globally, this is most likely not wanted.\n\n\n", stacklevel=0) if self._sample_data_how=="globally" else None

        
    def _log_to_wandb(self, step:int):
        """Logs losses to weights and biases."""
        if not self._log_wandb:
            return
        
        def _log_grad_norms(module: torch.nn.Module, prefix: str):
            """Logs L2 norms of gradients of a PyTorch module to wandb."""
            for name, param in module.named_parameters():
                if param.grad is not None:
                    metrics[f"grad_norms/{prefix}/{name}"] = param.grad.norm().item()

        metrics= {
            "train/left_actions/mean": float(self._losses['left_actions'][-1]),
            "train/regularizers/left_actions/lasso": float(self._losses['left_actions_tasks_reg'][-1]),

            "train/geometry/generator_span": float(self._losses['generator_span'][-1]),
            "train/geometry/generator_weights": float(self._losses['generator_weights'][-1]),
            "train/regularizers/generator/lasso": float(self._losses['generator_reg'][-1]),
            "train/geometry/reconstruction": float(self._losses['reconstruction_geo'][-1]),

            "train/symmetry/span": float(self._losses['symmetry'][-1]),
            "train/symmetry/reconstruction": float(self._losses['reconstruction_sym'][-1]),
            "train/symmetry/oracle_span": float(self._losses['oracle_symmetry_span'][-1]),
            "train/regularizers/symmetry": float(self._losses['symmetry_reg'][-1]),

            "diagnostics/cond_num_generator": float(self._diagnostics['cond_num_generator'][-1]),
            "diagnostics/frob_norm_generator": float(self._diagnostics['frob_norm_generator'][-1]),
            # "diagnostics/n_steps_gen": float(self.n_steps_gen),
        }

        if self._diagnostics["encoder_loss_oracle_sym"] is not None and self._diagnostics["decoder_loss_oracle_sym"] is not None:
            metrics["diagnostics/encoder_loss_oracle_sym"] = float(self._diagnostics['encoder_loss_oracle_sym'][-1])
            metrics["diagnostics/decoder_loss_oracle_sym"] = float(self._diagnostics['decoder_loss_oracle_sym'][-1])

        if self._diagnostics["encoder_loss_oracle_geo"] is not None and self._diagnostics["decoder_loss_oracle_geo"] is not None:
            metrics["diagnostics/encoder_loss_oracle_geo"] = float(self._diagnostics['encoder_loss_oracle_geo'][-1])
            metrics["diagnostics/decoder_loss_oracle_geo"] = float(self._diagnostics['decoder_loss_oracle_geo'][-1])

        task_losses= self._losses['left_actions_tasks'][-1]
        for idx_task in range(self._n_tasks-1):
            metrics[f"train/left_actions/tasks/task_idx={idx_task}"] = float(task_losses[idx_task])

        if self._log_wandb_gradients:
            _log_grad_norms(self.encoder_geo, "encoder_geo")
            _log_grad_norms(self.decoder_geo, "decoder_geo")
            _log_grad_norms(self.encoder_sym, "encoder_sym")
            _log_grad_norms(self.decoder_sym, "decoder_sym")
            _log_grad_norms(self.log_lgs, "log_lgs")
            _log_grad_norms(self.generator_S, "generator")
            _log_grad_norms(self.weights_lgs_to_gen, "weights_lgs_to_gen") if self._eval_span_how == "weights" else None

        wandb.log(metrics, step=step)













##############################################################################################################################
## Legacy code for symmetry discovery. Not used.
##############################################################################################################################








    # def compute_vec_vec_jacobian(self, f: callable, s: torch.tensor)->torch.tensor:
    #     """
    #     Compute a vectorized Jacobian of function f: n -> m over a batch of states s.
    #     Input: tensor s of shape (b,d,n) where both b and d are batch dimensions and n is the dimension of the data.
    #     Returns: tensor of shape (b,d,n,m)
    #     """
    #     return torch.vmap(torch.vmap(torch.func.jacrev(f)))(s)


    # def evaluate_symmetry(self, 
    #                      ps: torch.Tensor, 
    #                      generator:torch.Tensor,
    #                      encoder_sym:torch.nn.Module,
    #                      decoder_sym:torch.nn.Module,
    #                      track_loss: bool=True)->float:
    #     """
    #     Evaluates whether the generator is contained within the kernel distribution of the base task (expressed in symmetry and geometry charts).
    #     The following objects are frozen:
    #     - generator, as we are interested in finding a chart that symmetrizes f given the current geometry.
    #     - encoder_geo, decoder_geo, as these are used to represent the geometry.

    #     The following objects are trainable:
    #     - encoder_sym, decoder_sym, as these are used to represent the symmetry.
    #     """

    #     # Let the generator act on the points in latent space.
    #     tilde_ps=encoder_sym(ps)
    #     gen_tilde_ps = torch.einsum("dnm,bm->bdn", generator, tilde_ps)
    #     jac_decoder_sym=torch.vmap(torch.func.jacrev(decoder_sym))(tilde_ps)
    #     # jac_decoder_sym=torch.vmap(torch.vmap(torch.func.jacrev(decoder_sym)))(gen_tilde_ps) #TODO, changed this line to the one above
    #     # jac_decoder_sym=self.compute_vec_vec_jacobian(decoder_sym, gen_tilde_ps)
    #     gen_ps = torch.einsum("bmn, bdn->bdm", jac_decoder_sym, gen_tilde_ps)

    #     # Check symmetry, need to evaluate frame at base points.
    #     frame_ps=self.kernel_frame_base_task(ps)
    #     frame_ps=frame_ps.unsqueeze(0)
    #     frame_ps=frame_ps.view(1, -1, self.kernel_dim, self.ambient_dim_R)

    #     # Evaluate how far out of the kernel distribution the generator is.
    #     _, gen_into_frame = self._project_onto_vector_subspace(gen_ps, frame_ps)

    #     loss_symmetry= torch.linalg.vector_norm(gen_into_frame, dim=-1).mean(-1).mean()
    #     loss_reconstruction_sym= torch.linalg.vector_norm(ps - decoder_sym(encoder_sym(ps)), dim=-1).mean()
    #     loss_reg = self._lasso_coef_encoder_decoder * (self._l1_penalty(encoder_sym) + self._l1_penalty(decoder_sym))


    #     # Compare the current encoder to an oracle encoder, only used for debugging.
    #     if self._oracle_encoder_sym is not None and self._oracle_decoder_sym is not None:
    #         with torch.no_grad():
    #             encoder_loss_oracle_sym = torch.norm(self._oracle_encoder_sym(ps) - encoder_sym(ps), dim=-1).sum(0)
    #             decoder_loss_oracle_sym = torch.norm(self._oracle_decoder_sym(ps) - decoder_sym(ps), dim=-1).sum(0)
    #         if track_loss:
    #             self._diagnostics["encoder_loss_oracle_sym"].append(encoder_loss_oracle_sym.detach().cpu().numpy())
    #             self._diagnostics["decoder_loss_oracle_sym"].append(decoder_loss_oracle_sym.detach().cpu().numpy())

    #     if self._oracle_kernel is not None:
    #         # If oracle frames are provided, compute the symmetry loss with respect to the oracle frames.
    #         with torch.no_grad():

    #             def create_linear_mesh(num_points, x_lower, x_upper, y_lower, y_upper):
    #                 x = torch.linspace(x_lower, x_upper, num_points)
    #                 y = torch.linspace(y_lower, y_upper, num_points)
    #                 grid_x, grid_y = torch.meshgrid(x, y)
    #                 mesh_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    #                 return mesh_points
                
    #             #Warning, this is only meaningful for our circle
    #             #example if the base task is in the top left.
    #             ps = create_linear_mesh(num_points=10, x_lower=-1, x_upper=0, y_lower=0, y_upper=1) 

    #             tilde_ps=encoder_sym(ps)
    #             gen_tilde_ps = torch.einsum("dnm,bm->bdn", generator, tilde_ps)
    #             jac_decoder_sym=torch.vmap(torch.func.jacrev(decoder_sym))(tilde_ps)
    #             gen_ps = torch.einsum("bmn, bdn->bdm", jac_decoder_sym, gen_tilde_ps)

    #             oracle_frame_ps=self._oracle_kernel[self.base_task_index](ps)
    #             oracle_frame_ps=oracle_frame_ps.unsqueeze(0)

    #             _, gen_into_oracle_frame = self._project_onto_vector_subspace(gen_ps, oracle_frame_ps)
    #             oracle_loss_symmetry= torch.linalg.vector_norm(gen_into_oracle_frame, dim=-1).mean(-1).mean()
    #             self._losses["oracle_symmetry_span"].append(oracle_loss_symmetry.detach().cpu().numpy())
    #             del oracle_frame_ps, gen_into_oracle_frame, oracle_loss_symmetry, ps, tilde_ps, gen_tilde_ps, jac_decoder_sym, gen_ps

    #     if track_loss:
    #         self._losses["symmetry"].append(loss_symmetry.detach().cpu().numpy())
    #         self._losses["reconstruction_sym"].append(loss_reconstruction_sym.detach().cpu().numpy())
    #         self._losses["symmetry_reg"].append(loss_reg.detach().cpu().numpy())


    #     return loss_symmetry+loss_reconstruction_sym+loss_reg


    # def take_step_sym(self):
    #     """
    #     Steps all symmetry variables: generator and chart.
    #     """
    #     ps = self.sample_data(task_idx=self.base_task_index, sample_data_how=self._sample_data_how)

    #     for p in self.generator_S.parameters(): p.requires_grad = True
    #     for p in self.encoder_sym.parameters(): p.requires_grad = True
    #     for p in self.decoder_sym.parameters(): p.requires_grad = True

    #     self.optimizer_generator_S.zero_grad()
    #     self.optim_encoder_sym.zero_grad()
    #     self.optim_decoder_sym.zero_grad()

    #     generator_normed = self.generator_S.param / torch.linalg.matrix_norm(self.generator_S.param)
    #     loss_symmetry = self.evaluate_symmetry(ps=ps, 
    #                                            generator=generator_normed, 
    #                                            encoder_sym=self.encoder_sym,
    #                                            decoder_sym=self.decoder_sym)
    #     loss_symmetry.backward()            
    #     self.optim_encoder_sym.step()
    #     self.optim_decoder_sym.step()
    #     self.optimizer_generator_S.step()


    # def _stack_samples(self):
    #     """Stacks samples from all tasks into a single tensor."""
    #     _n_samples_per_task, ambient_dim = self.tasks_data_R[0].shape
    #     ps = torch.empty([self._n_tasks, _n_samples_per_task, ambient_dim], dtype=torch.float32)
    #     for i, task_ps in enumerate(self._n_tasks):
    #         ps[i] = task_ps
    #     self.all_ps=ps.reshape([-1, ambient_dim])


    # def sample_data(self, task_idx:int, sample_data_how: Literal["uniform_replay", "uniform_manifold"]) -> torch.Tensor:
    #     """Samples a batch of data from task task_idx either uniformly from the replay buffer or uniformly from the support of the replay buffer."""

    #     if sample_data_how == "uniform_replay":
    #         indices=torch.randint(0, self._n_samples, (self.batch_size,))
    #         return self.tasks_data_R[task_idx][indices]
        
    #     elif sample_data_how == "uniform_manifold":
    #         assert task_idx==self.base_task_index, "Uniform manifold method only supported for base task. Compute kernel density for other tasks but recall using pushforward."
    #         indices = torch.multinomial(self.uniform_replay_weights, num_samples=self.batch_size, replacement=False)
    #         return self.tasks_data_R[task_idx][indices]

    #     elif sample_data_how =="globally":
    #         return torch.randn(size=(self.batch_size, self.ambient_dim_R))

    #     else:
    #         raise KeyError(f"Supported sampling modes: ['uniform_replay', 'uniform_manifold'], you provided: {sample_data_how}")