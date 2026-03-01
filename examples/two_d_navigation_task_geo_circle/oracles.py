import torch
import logging
from src.utils import DenseNN

def rotation_vector_field(p_batch: torch.tensor, center)->torch.tensor:
    """Returns kernel samples at batched points p from a task."""
    _generator=torch.tensor([[0, -1], [1,0]], requires_grad=False, dtype=torch.float32).unsqueeze(0)
    projected_state = p_batch-center
    gradients = torch.einsum("dmn, bn->bdm", _generator, projected_state)
    norm_gradients = gradients.norm(dim=-1, keepdim=True)
    return gradients/norm_gradients

def make_2d_navigation_oracles(goal_locations,
                               save_dir_base: str):
    """Creates all oracles for the 2D navigation task."""
    
    base_goal_location = goal_locations[0]["goal"]
    # Define oracle charts and generator, only used for debugging.
    ORACLE_GENERATOR=torch.tensor([[0,-1], [1,0]], dtype=torch.float32, requires_grad=False).unsqueeze(0)
    ORACLE_ENCODER_GEO, ORACLE_DECODER_GEO, ORACLE_ENCODER_SYM, ORACLE_DECODER_SYM=DenseNN([2,2]), DenseNN([2,2]), DenseNN([2,2]), DenseNN([2,2])

    def set_affine_weights(network, weight, bias):
        """Sets the weights of an affine function represented as a neural network."""
        linear_layer=network.net[0]
        linear_layer.weight.data = weight
        linear_layer.bias.data = bias
        return linear_layer

    with torch.no_grad():
        ORACLE_DECODER_GEO = set_affine_weights(ORACLE_DECODER_GEO, torch.eye(2), torch.zeros(2))
        ORACLE_ENCODER_GEO = set_affine_weights(ORACLE_ENCODER_GEO, torch.eye(2), torch.zeros(2))
        ORACLE_ENCODER_SYM = set_affine_weights(ORACLE_ENCODER_SYM, torch.eye(2), -torch.tensor(base_goal_location, dtype=torch.float32))
        ORACLE_DECODER_SYM = set_affine_weights(ORACLE_DECODER_SYM, torch.eye(2), torch.tensor(base_goal_location, dtype=torch.float32))

    ORACLE_KERNELS = [lambda ps, c=torch.tensor(task['goal'], dtype=torch.float32): rotation_vector_field(ps, center=c) for task in goal_locations]
    
    # Oracle transition functions.
    def T(s, a):
        """Oracle transition function in the 2-D navigation task."""
        return s + a
    ORACLE_T_FNS = [T for _ in goal_locations]
    
    oracles = {
        'goal_locations': goal_locations,
        'generator': ORACLE_GENERATOR,
        'encoder_geo': ORACLE_ENCODER_GEO,
        'decoder_geo': ORACLE_DECODER_GEO,
        'encoder_sym': ORACLE_ENCODER_SYM,
        'decoder_sym': ORACLE_DECODER_SYM,
        'kernel': ORACLE_KERNELS,
        'T_fns': ORACLE_T_FNS,
    }

    torch.save({
        'goal_locations': goal_locations,
        'generator': ORACLE_GENERATOR,
        'encoder_geo': ORACLE_ENCODER_GEO.state_dict(),
        'decoder_geo': ORACLE_DECODER_GEO.state_dict(),
        'encoder_sym': ORACLE_ENCODER_SYM.state_dict(),
        'decoder_sym': ORACLE_DECODER_SYM.state_dict(),
    }, f"{save_dir_base}/oracles.pt")
    logging.info(f"Saved oracles to {save_dir_base}/oracles.pt.")

    return oracles