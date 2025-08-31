import os
import torch


# Define path(s) for compressed files
def compressed_files(file_name, dir):
    head_path = os.path.join(dir, file_name+'.h.bin')
    bones_path = os.path.join(dir, file_name+'.b.bin')
    skin_path = os.path.join(dir, file_name+'.s.bin')
    cache_path = os.path.join(dir, '__cache__.ply')
    return head_path, bones_path, skin_path, cache_path

# Remap old checkpoint keys to new model keys
def remap_state_dict(old_state_dict, model):
    """
    Remap old checkpoint keys to new model keys by matching shape and partial name.
    Returns a new state_dict compatible with the new model.
    """
    new_state_dict = {}
    model_state_dict = model.state_dict()
    used_keys = set()
    unmatched_keys = []
    for new_key, new_tensor in model_state_dict.items():
        # Try exact match first
        if new_key in old_state_dict and old_state_dict[new_key].shape == new_tensor.shape:
            new_state_dict[new_key] = old_state_dict[new_key]
            used_keys.add(new_key)
            continue
        # Try partial match by suffix
        for old_key, old_tensor in old_state_dict.items():
            if old_key.endswith(new_key.split('.')[-1]) and old_tensor.shape == new_tensor.shape and old_key not in used_keys:
                new_state_dict[new_key] = old_tensor
                used_keys.add(old_key)
                break
        else:
            unmatched_keys.append(new_key)
            new_state_dict[new_key] = new_tensor
    if unmatched_keys:
        raise KeyError(f"Could not map the following keys from old checkpoint: {unmatched_keys}")
    return new_state_dict
