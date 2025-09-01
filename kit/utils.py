import os
import torch

# Get mapping from old to new state dict
def remap_old_to_new(old_state):
    new_state = {}
    for k, v in old_state.items():
        # Explicit pt_block remapping
        if k.startswith("pt_block.attn."):
            new_key = "encoder." + k
            new_state[new_key] = v

        # fea_squeeze
        elif k.startswith("fea_squeeze"):
            new_key = "encoder." + k
            new_state[new_key] = v

        # Encoder: awds
        elif k.startswith("awds"):
            new_state["encoder." + k] = v

        # Entropy model
        elif k.startswith("dwem") or k.startswith("dw_build"):
            new_state["entropy_model." + k] = v

        # Decoder
        elif k.startswith("fea_stretch") or k.startswith("dwus"):
            new_state["decoder." + k] = v

        else:
            print(f" Skipping unmapped key: {k}")
    return new_state

def load_legacy_checkpoint(new_model, ckpt_path, device="cpu"):
    old_state = torch.load(ckpt_path, map_location=device)
    remapped_state = remap_old_to_new(old_state)

    missing, unexpected = new_model.load_state_dict(remapped_state, strict=False)

    print(" Loaded legacy checkpoint (with remapping).")
    if missing or unexpected:
        print(" Missing keys (new layers not in old ckpt):", missing)
        print(" Unexpected keys (in ckpt but unused):", unexpected)

    return new_model

# Define path(s) for compressed files
def compressed_files(file_name, dir):
    head_path = os.path.join(dir, file_name+'.h.bin')
    bones_path = os.path.join(dir, file_name+'.b.bin')
    skin_path = os.path.join(dir, file_name+'.s.bin')
    cache_path = os.path.join(dir, '__cache__.ply')
    return head_path, bones_path, skin_path, cache_path
