from typing import Literal, Optional, List
import logging
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import os

import torch
import numpy as np

from src.utils import SingleLayerNet
from src.learning.aa_policy_training.utils import load_replay_buffer
from src.learning.bb_symmetry.kernel_approx import KernelFrameEstimator


def compute_and_save_kernel(dir: str,
                                  args):
    """
    Learn a frame of a kernel distribution from a replay buffer by first learning pointwise kernel bases and then training a neural network to approximate the frame.

    Returns:
    neural_kernel: FrameNet
        A neural network that approximates the frame of the kernel distribution.
    """

    replay_buffer_name:str=dir+"/replay_buffer.pkl"
    n_steps=int(args.n_steps_train_pis/args.n_envs)
    replay_buffer = load_replay_buffer(replay_buffer_name, N_steps=n_steps)
    states = replay_buffer["observations"]
    actions = replay_buffer["actions"]
    rewards = replay_buffer["rewards"]
    s_primes = replay_buffer["next_observations"]

    if args.which_data_R == ["observations", "actions"]:
        data_R = torch.hstack([states, actions])
    elif args.which_data_R == ["observations"]:
        data_R = states
    elif args.which_data_R == ["actions"]:
        data_R = actions
    elif args.which_data_R == ["next_observations"]:
        data_R = s_primes
    else:
        raise ValueError(f"which_data_R {args.which_data_R} not recognized.")
    
    if args.which_data_T == ["observations", "actions"]:
        data_T = torch.hstack([states, actions])
    elif args.which_data_T == ["observations"]:
        data_T = states
    elif args.which_data_T == ["actions"]:
        data_T = actions
    else:
        raise ValueError(f"which_data_T {args.which_data_T} not recognized.")


    logging.info("Computing frames...")
    frame_estimator = KernelFrameEstimator(
                                        states=states,
                                        actions=actions,
                                        rewards=rewards,
                                        s_primes=s_primes,        
                                        kernel_dim_R=args.kernel_dim,
                                        epsilon_ball=args.epsilon_ball,
                                        data_R = data_R,
                                        epsilon_level_set=args.epsilon_level_set,
                                        n_neighbors_in_level_set=args.n_neighbors_in_level_set,)
    kernel_basis_vectors = frame_estimator.compute()


    # Convert to pytorch tensor, only keep those samples where we get a reliable estimate of both kernels.
    torch.save(data_R, dir+"/all_data_R.pt") #This is used to initialze L_g and K_g in the final learning problem. If we use the later
    torch.save(data_T, dir+"/all_data_T.pt") #truncated version, the data of different tasks will have different shapes.
    idxs = list(set(kernel_basis_vectors.keys()))
    pointwise_kernel_frames = torch.stack([kernel_basis_vectors[i] for i in idxs], dim=0)
    # The kernel frames are stored as (n_samples, ambient_dim, kernel_dim), but we want (n_samples, kernel_dim, ambient_dim) for
    # the hereditary geometry discovery.
    pointwise_kernel_frames = pointwise_kernel_frames.permute(*range(pointwise_kernel_frames.ndim - 2), -1, -2)
    states = states[idxs]
    actions = actions[idxs]
    rewards = rewards[idxs]
    s_primes = s_primes[idxs]
    data_R = data_R[idxs]
    data_T = data_T[idxs]

    neural_kernel, _ = train_neural_kernel(
                        data_R=data_R,
                        kernel_bases=pointwise_kernel_frames,
                        lasso_coef=args.lasso_coef_kernel,
                        sample_data_how=args.sample_data_how,
                        temperature=args.temperature,
                        epochs=args.n_epochs_neural_kernel,
                        k=args.neighbors_uniform_sampling,
                        batch_size=args.batch_size_neural_kernel,
                        )
  
    torch.save(states, dir+"/states.pt")
    torch.save(actions, dir+"/actions.pt")
    torch.save(rewards, dir+"/rewards.pt")
    torch.save(s_primes, dir+"/s_primes.pt")
    torch.save(data_R, dir+"/data_R.pt")
    torch.save(data_T, dir+"/data_T.pt")
    torch.save(pointwise_kernel_frames, dir+"/kernel_samples.pt")
    torch.save(neural_kernel.state_dict(), dir+f"/neural_kernel.pt")


    return neural_kernel    

def train_neural_kernel(data_R: torch.Tensor,
                        kernel_bases: torch.Tensor,
                        
                        lasso_coef: float,
                        sample_data_how: Literal["uniform_replay", "uniform_manifold"],
                        temperature: Optional[float]=None,
                        k: Optional[int]=None,

                        epochs: int=2_000,
                        batch_size: int = 64,
                        ):
    
    
    """
    Learn two frames of the kernel distribution from samples using a dense neural network.
    
    Args:
    ps: torch.Tensor of shape (n_samples, ambient_dim)
    kernel_bases: torch.Tensor of shape (n_bases, kernel_dim, ambient_dim)
    """
    assert sample_data_how in ["uniform_replay", "uniform_manifold"], "Unsupported sampling mode. Choose 'uniform_replay' or 'uniform_manifold'."
    if sample_data_how == "uniform_manifold":
        assert temperature is not None, "Temperature must be provided for uniform manifold sampling."

    n_samples, kernel_dim, ambient_dim_kernel = kernel_bases.shape
    kernel_net = SingleLayerNet(ambient_dim_kernel, kernel_dim)
    kernel_optimizer = torch.optim.Adam(kernel_net.parameters(), lr=1e-3)
    kernel_losses= []
    progress_bar_kernel = tqdm(range(epochs), desc=f"Train kernel w/ sampling {sample_data_how} and β={temperature}.", unit="epoch")

    # 1. Set up sampling weights.
    if sample_data_how == "uniform_manifold":
        weights_data_R = compute_boltzman_sampling_weights(data_R, temperature, k)
    else:
        weights_data_R = torch.ones(n_samples) / n_samples

    # Train the neural kernel distribution.
    for epoch in progress_bar_kernel:
        idxs = torch.multinomial(weights_data_R, num_samples=batch_size, replacement=False)
        s_and_as_batch= data_R[idxs]
        bases_batch= kernel_bases[idxs]

        kernel_optimizer.zero_grad()
        frame_hat = kernel_net(s_and_as_batch)

        # Orientation and magnitude invariant loss. #In higher dimensions this also needs to be rotation invariant, e.g. invariant w.r.t. the span.
        loss_kernel = -torch.mean(torch.mean(torch.abs(torch.cosine_similarity(frame_hat, bases_batch, dim=-1)), dim=-1), dim=-1) #Use absolute of negative cosine similarity as orientation does not signify.
        loss_lasso = torch.mean(torch.abs(frame_hat))
        loss = loss_kernel + lasso_coef * loss_lasso
        loss.backward()
        kernel_optimizer.step()

        kernel_losses.append(loss.item())
        progress_bar_kernel.set_description(f"Training kernel — loss={loss.item():.4f} (should be -1)")

    return kernel_net, kernel_losses



def compute_boltzman_sampling_weights(data: torch.tensor, temperature: float, k: int) -> torch.tensor:
    """Compute Boltzman sampling weights based on a k-nearest neighbor density estimation."""

    logging.info("Initializing uniform manifold sampling...")

    data_np = data.cpu().numpy()
    _, d = data_np.shape
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(data_np)
    dists, _ = nbrs.kneighbors(data_np)
    rk = dists[:, -1]
    log_dens = -d * np.log(rk + 1e-12)
    log_dens_t = torch.tensor(log_dens, dtype=torch.float32)
    log_weights = -temperature * log_dens_t
    log_weights = log_weights - log_weights.max()

    weights_unnorm = torch.exp(log_weights)
    weights = weights_unnorm / weights_unnorm.sum()

    logging.info("Initialized uniform manifold sampling.")

    return torch.tensor(weights)