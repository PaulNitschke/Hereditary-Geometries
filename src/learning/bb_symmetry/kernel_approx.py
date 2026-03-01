from typing import Tuple, Dict, Optional
import warnings
import logging
logging.basicConfig(level=logging.INFO)

from tqdm import tqdm
import numpy as np
import pickle
from scipy.spatial import KDTree
import torch

from constants import DTYPE

class KernelFrameEstimator():

    def __init__(self,
                states: torch.Tensor,
                actions: torch.tensor,
                rewards: torch.tensor,
                s_primes: torch.Tensor,
                data_R: torch.Tensor,
                kernel_dim_R: int,
                epsilon_ball: Optional[float] = None,
                epsilon_level_set: Optional[float] = None,
                use_relative_epsilon_level_set: bool = False,
                n_neighbors_in_level_set: int = None):
        
        """Computes pointwise bases (a frame) of a kernel of a function f: M \rightarrow N by performing a first degree Taylor expansion of f.
        
        Args:
        - ps: torch.tensor of shape (n_samples, |M|).
        - kernel_dim: int, dimension of the kernel
        - ns: torch.tensor of shape (n_samples, |N|). Not required for inference only.
        - epsilon_ball: float, radius of ball on which we Taylor approximate f. Not required for inference only.
        - epsilon_level_set: float, up to what tolerance are p,p' in the same level set, e.g. |f(p)-f(p')|< epsilon_level_set. Generally, epsilon_level_set should be smaller than epsilon_ball
                                Not required for inference only.
        - n_neighbors_in_level_set: int, the minimum number of neighbors in the level set of a point p to compute a basis.
        """

        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.s_primes = s_primes
        self.data_R = data_R

        self.kernel_dim_R = kernel_dim_R
        self.epsilon_ball = epsilon_ball
        self.epsilon_level_set = epsilon_level_set
        self.use_relative_epsilon_level_set = use_relative_epsilon_level_set
        self.n_neighbors_in_level_set = n_neighbors_in_level_set

        self.dim_S = self.states.shape[1]
        self.dim_A = self.actions.shape[1]

        self.pointwise_frame = {}
        self._finished_setup_evaluation=False


    def compute(self) -> dict[int, torch.tensor]:
        """
        Computes a pointwise frame of the kernel distribution.
        Returns:
        - dict:
            - keys: indices of each data sample where the kernel frame could be computed.
            - values: basis vectors of shape (kernel_dim, |M])
        """
        assert self.epsilon_ball is not None, "Kernel approximation requires a radius of the ball on which we Taylor approximate f."
        assert self.epsilon_level_set is not None, "Kernel approximation requires a tolerance level for the level set of f."

        warnings.warn("TODO: Dimension of kernel should be actively inferred, not passed as an argument.")

        # Compute pointwise samples from the kernel distribution of R.
        kernel_vectors, _ = self._compute_kernel_samples(self.data_R, self.rewards, self.epsilon_level_set)
        kernel_basis_vectors = self._compute_pointwise_basis(kernel_vectors=kernel_vectors, kernel_dim=self.kernel_dim_R)

        return kernel_basis_vectors

    

    def save(self, file_name:str):
        """Saves the frame samples to file."""
        with open(file_name, 'wb') as f:
            pickle.dump(self.pointwise_frame, f)

    
    def set_frame(self, frame: dict[int, torch.tensor]):
        """Sets the pointwise frame of the kernel distribution and initializes the kernel evaluation."""
        self.pointwise_frame = frame
        self.setup_evaluation()


    def setup_evaluation(self):
        """Sets up the sampling from the kernel distribution by building an approximate k-nearest neighbor graph."""
        self.known_idx = torch.tensor(list(self.pointwise_frame.keys()))
        self.known_ps = self.ps[self.known_idx]
        self.known_frames = torch.stack([self.pointwise_frame[int(i)] for i in self.known_idx])
        self._finished_setup_evaluation=True
        logging.info("Setup kernel frame evaluation.")


    def _compute_neighborhood(self, data, epsilon) -> list:
        """Computes the neighborhood of each point in data.
        Args:
            data: torch.Tensor of shape (n_samples, self.dim_S_and_A)
            epsilon: float"

        Returns:
            list of lists of integers, where the ith element is the list of indices of the neighbors of the ith point in data.
        """
        tree = KDTree(data.numpy())
        return tree.query_ball_tree(tree, epsilon)


    def _compute_kernel_samples(self, 
                                data_R, 
                                rewards, 
                                epsilon_level_set) -> Tuple[Dict, Dict]:
        """
        Computes a pointwise approximation of samples from the kernel of $f$.

        Args:
            neighbors: list of lists of integers, where the ith element is the list of indices of the neighbors of the ith point in data.
            ns: torch.Tensor of shape (n_samples,), the readout of f at each point in data.
            epsilon_level_set: float, tolerance level for the level set.

        Returns:
            kernel_vectors: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, self.dim_S_and_A) containing a sample from the kernel distribution
            self.local_level_set: TODO
        """
        neighbors = self._compute_neighborhood(data_R, self.epsilon_ball)

        # a) Approximate which samples are (i) close to a given sample and (ii) belong to the same level.
        self.local_level_set = {}
        for idx, y in tqdm(enumerate(rewards), desc="Locate samples in local level set...", total=len(rewards)):

            x_neighbors_idxs = np.array(neighbors[idx])
            x_neighbors_idxs = x_neighbors_idxs[x_neighbors_idxs != idx] #remove the point itself

            if self.use_relative_epsilon_level_set:
                x_neighbors_level_set_membership = np.array(torch.abs((rewards[x_neighbors_idxs] - y)/y) < epsilon_level_set)
            else:
                x_neighbors_level_set_membership = np.array(torch.abs(rewards[x_neighbors_idxs] - y) < epsilon_level_set)
            x_neighbors_level_set_idxs = x_neighbors_idxs[x_neighbors_level_set_membership]

            self.local_level_set[idx] = x_neighbors_level_set_idxs

        # b) Approximate the exponential map between samples in the same level set and in an \epsilon-ball via a linear approximation.
        kernel_vectors = {}
        for idx_sample, sample in tqdm(enumerate(data_R), desc="Compute pointwise kernel samples...", total=len(data_R)):
            kernel_vectors[idx_sample] = torch.zeros((len(self.local_level_set[idx_sample]), self.dim_S + self.dim_A), dtype=DTYPE)

            level_set_neighbors = self.local_level_set[idx_sample]
            diffs = data_R[level_set_neighbors] - sample


            _norms= torch.linalg.norm(diffs, dim=1, keepdim=True)
            diffs = torch.where(_norms > 0, diffs / _norms, diffs)
            kernel_vectors[idx_sample] = diffs

        return kernel_vectors, self.local_level_set


    def _compute_pointwise_basis(self,
                                 kernel_vectors: dict[int, torch.tensor],
                                kernel_dim: int) -> dict[int, torch.tensor]:
        """
        Computes a 1-D pointwise basis of the kernel distribution at each sample if there exists at least one non-trivial tangent vector in the kernel.

        Args:
            kernel_vectors: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, self.dim_S_and_A) containing a sample from the kernel distribution
            kernel_dim: int, the dimension of the Kernel. TODO: this should be actively inferred.

        Returns:
            basis: dictonary of length (n_samples,), each key is the index of a sample and the value is a tensor of shape (n_kernel_vectors, self.dim_S_and_A) containing a basis of the kernel distribution at the
        """
        nums = np.arange(0, self.n_neighbors_in_level_set)
        count_n_tangents={}
        for num in nums:
            count_n_tangents[f"{num}"]=0
        count_n_tangents[f">={self.n_neighbors_in_level_set}"]=0

        n_samples = len(kernel_vectors.keys())
        for idx_sample in tqdm(kernel_vectors.keys(), desc="Compute Point-Wise Bases via PCA..."):

            kernel_vectors_point = kernel_vectors[idx_sample]
            n_kernel_vectors_point= len(kernel_vectors_point)
            if n_kernel_vectors_point < self.n_neighbors_in_level_set:
                count_n_tangents[f"{n_kernel_vectors_point}"]+=1
            else:

                _, _, _basis_vector = torch.pca_lowrank(kernel_vectors_point, q=kernel_dim, center=False)
                self.pointwise_frame[idx_sample] = _basis_vector
                count_n_tangents[f">={self.n_neighbors_in_level_set}"]+=1

        info=[f"{num} tangent vectors for {round(100 * count_n_tangents[f'{num}'] / n_samples, 2)}% of samples" for num in nums]
        info.append(f">={self.n_neighbors_in_level_set} tangent vectors for {round(100 * count_n_tangents[f'>={self.n_neighbors_in_level_set}'] / n_samples, 2)}% of samples. Discarding all other tangent vectors.")
        logging.info("\n".join(info))

        return self.pointwise_frame