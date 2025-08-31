import torch
import torch.nn as nn

import module
import kit.op as op


# Main Pointsoup model (Pointsoup)
class Pointsoup(nn.Module):
    """
    Main Pointsoup model. Composes encoder, entropy model, and decoder.
    Args:
        k (int): Dilated window size.
        channel (int): Feature channel size.
        bottleneck_channel (int): Bottleneck channel size.
    """
    def __init__(self, k, channel, bottleneck_channel):
        super().__init__()
        # Encoder
        self.encoder = Encoder(channel, bottleneck_channel)
        # Entropy model
        self.entropy_model = EntropyModel(k, channel, bottleneck_channel)
        # Decoder 
        self.decoder = Decoder(channel, bottleneck_channel)

    def forward(self, batch_x, K):
        N = batch_x.shape[1]
        # Encoding
        compact_fea, bones = self.encoder(batch_x, K)
        # Entropy Modeling
        quantized_compact_fea, dilated_idx, dilated_windows, bitrate = self.entropy_model(compact_fea, bones, N)
        # Decoding
        rec_batch_x = self.decoder(quantized_compact_fea, dilated_idx, dilated_windows, bones, K)
        return rec_batch_x, bitrate


# Encoder class (Pointsoup)
class Encoder(nn.Module):
    def __init__(self, channel, bottleneck_channel):
        super().__init__()
        self.awds = module.AWDS(channel=channel)
        self.fea_squeeze = nn.Linear(channel, bottleneck_channel)

    def forward(self, batch_x, K):
        # Adaptive Window Downsampling
        skin_fea, bones = self.awds(batch_x, K)
        # Feature Squeezing
        compact_fea = self.fea_squeeze(skin_fea)
        return compact_fea, bones
    

# Entropy Model class (Pointsoup)
class EntropyModel(nn.Module):
    def __init__(self, k, channel, bottleneck_channel):
        super().__init__()
        self.dw_build = module.DWBuild(k=k)
        self.dwem = module.DWEM(channel=channel, bottleneck_channel=bottleneck_channel)

    def forward(self, compact_fea, bones, N):
        # Quantization of compact features
        quantized_cf = compact_fea + torch.nn.init.uniform_(torch.zeros_like(compact_fea), -0.5, 0.5)
        # Dilated Convolution - dilated idx, windows
        d_idx, d_windows = self.dw_build(bones)
        # Feature Extraction
        mu, sigma = self.dwem(d_windows)
        # Bitrate Estimation
        bitrate, _ = op.feature_probs_based_mu_sigma(quantized_cf, mu, sigma)
        bitrate = bitrate / N
        return quantized_cf, d_idx, d_windows, bitrate


# Decoder class (Pointsoup)
class Decoder(nn.Module):
    def __init__(self, channel, bottleneck_channel):
        super().__init__()
        self.fea_stretch = nn.Linear(bottleneck_channel, channel)
        self.dwus = module.DWUS(channel=channel, fold_channel=8, R_max=256, r=4)

    def forward(self, quantized_cf, d_idx,  d_windows, bones, K):
        # Feature Stretching
        rec_skin_fea = self.fea_stretch(quantized_cf)
        # Dilated Window Upsampling
        rec_batch_x = self.dwus(rec_skin_fea, bones, d_windows, d_idx, K)
        return rec_batch_x


# Pointsoup variant with Encoder updated to use SimpleSelfAttention
# -----------------------------------------------------------------
# The main difference between Pointsoup and PointsoupSelfAttention is:
# - Pointsoup uses the standard Encoder (AWDS + feature squeeze).
# - PointsoupSelfAttention inherits Pointsoup but replaces the encoder with EncoderUpdated,
#   which adds a SimpleSelfAttention block after AWDS for global feature correlation.
# This allows PointsoupSelfAttention to model more complex relationships in the point cloud.

class PointsoupSelfAttention(Pointsoup):
    """
    Pointsoup model with EncoderSelfAttention (SimpleSelfAttention).
    Args:
        k (int): Dilated window size.
        channel (int): Feature channel size.
        bottleneck_channel (int): Bottleneck channel size.
    """
    def __init__(self, k, channel, bottleneck_channel):
        super().__init__(k, channel, bottleneck_channel)
        self.encoder = EncoderSelfAttention(channel, bottleneck_channel)


# Encoder class: Self-Attention variant
class EncoderSelfAttention(nn.Module):
    def __init__(self, channel, bottleneck_channel):
        super().__init__()
        self.awds = module.AWDS(channel=channel)
        self.pt_block = module.SimpleSelfAttention(channel)
        self.fea_squeeze = nn.Linear(channel, bottleneck_channel)

    def forward(self, batch_x, K):
        # Adaptive Window Downsampling
        skin_fea, bones = self.awds(batch_x, K)
        # Point-wise Attention
        skin_fea = self.pt_block(skin_fea)
        # Feature Squeezing
        compact_fea = self.fea_squeeze(skin_fea)
        return compact_fea, bones


# Model type selection:
#   "pointsoup"    -> Pointsoup (standard)
#   "pointsoup_sa" -> PointsoupSelfAttention (with self-attention encoder)
def model(ctx, model_type="pointsoup"):
    if model_type == "pointsoup_sa":
        model = PointsoupSelfAttention(k=ctx.dilated_window_size,
                            channel=ctx.channel, 
                            bottleneck_channel=ctx.bottleneck_channel)
        print("[TRAIN] Using \033[1;32mPointsoup Self-Attention\033[0m model.")
    else:
        model = Pointsoup(k=ctx.dilated_window_size,
                            channel=ctx.channel, 
                            bottleneck_channel=ctx.bottleneck_channel)
        print("[TRAIN] Using \033[1;34mPointsoup\033[0m model.")
    return model
