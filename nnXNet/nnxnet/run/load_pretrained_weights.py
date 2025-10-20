import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch._dynamo import OptimizedModule
from typing import Union

def load_pretrained_weights(network, fname: Union[dict, str], verbose: bool = False):
    """
    Loads weights from a pretrained model, matching keys by name and shape.

    Segmentation layers and other specified layers are skipped. This function is more
    robust and will not fail if the pretrained model contains keys that are not present
    in the current network, or vice versa.
    """
    if dist.is_initialized():
        saved_model = torch.load(fname, map_location=torch.device('cuda', dist.get_rank()), weights_only=False)
    else:
        saved_model = torch.load(fname, weights_only=False)

    pretrained_dict = saved_model['network_weights']

    # Handle common DDP and torch.optim prefixes
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        key = k
        if key.startswith("module.backbone."):
            key = key.replace("module.backbone.", "")
        if key.startswith("module."):
            key = key[7:]
        if key.startswith("_orig_mod."):
            key = key[10:]
        new_state_dict[key] = v

    pretrained_dict = new_state_dict

    # Define keys to skip from the pretrained model. You can customize this list.
    skip_strings = [] #['encoder'] #['.seg_layers.']

    # Get the target network's state dictionary
    if isinstance(network, DDP):
        mod = network.module
    elif isinstance(network, OptimizedModule):
        mod = network._orig_mod
    else:
        mod = network
    
    model_dict = mod.state_dict()

    # Filter pretrained dictionary to only include keys that match both name and shape
    matched_state_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and
           model_dict[k].shape == v.shape and
           not any(skip_str in k for skip_str in skip_strings)
    }

    # Load the filtered state dictionary
    if verbose:
        print("################### Loading pretrained weights from file ", fname, '###################')
        print("Below is the list of overlapping blocks in pretrained model and current network:")
        for key, value in matched_state_dict.items():
            print(f"Loading key: {key}, shape: {value.shape}")
        print("################### Done ###################")

    # Use strict=False to allow for partial loading
    mod.load_state_dict(matched_state_dict, strict=False)