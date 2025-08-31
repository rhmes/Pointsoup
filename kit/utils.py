import os
import torch


# Define path(s) for compressed files
def compressed_files(file_name, dir):
    head_path = os.path.join(dir, file_name+'.h.bin')
    bones_path = os.path.join(dir, file_name+'.b.bin')
    skin_path = os.path.join(dir, file_name+'.s.bin')
    cache_path = os.path.join(dir, '__cache__.ply')
    return head_path, bones_path, skin_path, cache_path
