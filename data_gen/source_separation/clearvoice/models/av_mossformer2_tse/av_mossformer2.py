import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import math

from .mossformer.utils.one_path_flash_fsmn import Dual_Path_Model, SBFLASHBlock_DualA
from .visual_frontend import Visual_encoder

EPS = 1e-8

class Mossformer(nn.Module):
    def __init__(self, args):
        super(Mossformer, self).__init__()
        
        N, L, = args.network_audio.encoder_out_nchannels, args.network_audio.encoder_kernel_size

        self.encoder = Encoder(L, N)
        self.separator = Separator(args)
        self.decoder = Decoder(args, N, L)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, mixture, visual):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        mixture_w = self.encoder(mixture)
        est_mask = self.separator(mixture_w, visual)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)
        est_source = F.pad(est_source, (0, T_origin - T_conv))
        return est_source

class Encoder(nn.Module):
    def __init__(self, L, N):
        super(Encoder, self).__init__()
        self.L, self.N = L, N
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, args, N, L):
        super(Decoder, self).__init__()
        self.N, self.L, self.args = N, L, args
        self.basis_signals = nn.Linear(N, L, bias=False)

    def forward(self, mixture_w, est_mask):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        est_source = mixture_w * est_mask
        est_source = torch.transpose(est_source, 2, 1) # [M,  K, N]
        est_source = self.basis_signals(est_source)  # [M,  K, L]
        est_source = overlap_and_add(est_source, self.L//2) # M x C x T
        return est_source




class Separator(nn.Module):
    def __init__(self, args):
        super(Separator, self).__init__()

        self.layer_norm = nn.GroupNorm(1, args.network_audio.encoder_out_nchannels, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(args.network_audio.encoder_out_nchannels, args.network_audio.encoder_out_nchannels, 1, bias=False)

        # mossformer 2
        intra_model = SBFLASHBlock_DualA(
            num_layers=args.network_audio.intra_numlayers,
            d_model=args.network_audio.encoder_out_nchannels,
            nhead=args.network_audio.intra_nhead,
            d_ffn=args.network_audio.intra_dffn,
            dropout=args.network_audio.intra_dropout,
            use_positional_encoding=args.network_audio.intra_use_positional,
            norm_before=args.network_audio.intra_norm_before
        )

        self.masknet = Dual_Path_Model(
            in_channels=args.network_audio.encoder_out_nchannels,
            out_channels=args.network_audio.encoder_out_nchannels,
            intra_model=intra_model,
            num_layers=args.network_audio.masknet_numlayers,
            norm=args.network_audio.masknet_norm,
            K=args.network_audio.masknet_chunksize,
            num_spks=args.network_audio.masknet_numspks,
            skip_around_intra=args.network_audio.masknet_extraskipconnection,
            linear_layer_after_inter_intra=args.network_audio.masknet_useextralinearlayer
        )

        # reference
        self.av_conv = nn.Conv1d(args.network_audio.encoder_out_nchannels+args.network_reference.emb_size, args.network_audio.encoder_out_nchannels, 1, bias=True)


    def forward(self, x, visual):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, D = x.size()

        x = self.layer_norm(x)
        x = self.bottleneck_conv1x1(x)


        visual = F.interpolate(visual, (D), mode='linear')
        x = torch.cat((x, visual),1)
        x  = self.av_conv(x)

        x = self.masknet(x)

        x = x.squeeze(0)

        return x



def overlap_and_add(signal, frame_step):
    """Reconstructs a signal from a framed representation.

    Adds potentially overlapping frames of a signal with shape
    `[..., frames, frame_length]`, offsetting subsequent frames by `frame_step`.
    The resulting tensor has shape `[..., output_size]` where

        output_size = (frames - 1) * frame_step + frame_length

    Args:
        signal: A [..., frames, frame_length] Tensor. All dimensions may be unknown, and rank must be at least 2.
        frame_step: An integer denoting overlap offsets. Must be less than or equal to frame_length.

    Returns:
        A Tensor with shape [..., output_size] containing the overlap-added frames of signal's inner-most two dimensions.
        output_size = (frames - 1) * frame_step + frame_length

    Based on https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/signal/python/ops/reconstruction_ops.py
    """
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes).unfold(0, subframes_per_frame, subframe_step)
    frame = signal.new_tensor(frame).long().cuda()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class av_mossformer2(nn.Module):
    def __init__(self, args):
        super(av_mossformer2, self).__init__()
        args.causal=0
        self.sep_network = Mossformer(args)
        self.ref_encoder = Visual_encoder(args)

    def forward(self, mixture, ref):
        ref = self.ref_encoder(ref)
        return self.sep_network(mixture, ref)


class AV_MossFormer2_TSE_16K(nn.Module):
    """MossFormer2 model wrapper for outside calling"""

    def __init__(self, args):
        super(AV_MossFormer2_TSE_16K, self).__init__()
        self.model = av_mossformer2(args)

    def forward(self, x):
        outputs = self.model(x)
        return outputs
