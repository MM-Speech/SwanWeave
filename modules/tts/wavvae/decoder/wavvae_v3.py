import argparse
import math
import torch
from torch import nn
import torch.nn.functional as F

from modules.tts.wavvae.decoder.diag_gaussian import DiagonalGaussianDistribution
from modules.tts.wavvae.decoder.feature_extractors import EncodecFeatures
from modules.tts.wavvae.decoder.latent2wav.modules import Generator, Upsample

class WavVAE_V3(nn.Module):
    def __init__(self, kp_nums=68, kp_channels=2,
                 latent_channels=64, hidden_size=512, use_firstframe_cond=False,
                 hparams=None):
        super().__init__()
        self.encoder = EncodecFeatures(dowmsamples=[6, 5, 4, 4, 2])
        self.proj_to_z = nn.Linear(512, 64)
        self.proj_to_decoder = nn.Linear(32, 320)

        config_path = hparams['melgan_config']
        args = argparse.Namespace()
        args.__dict__.update(config_path)
        self.latent_upsampler = Upsample(320, 4)
        self.decoder = Generator(
            input_size_=160, ngf=128, n_residual_layers=4,
            num_band=1, args=args, ratios=[5,4,4,3])

    def encode_latent(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b,t,latent_channel)
        return latent

    def encode(self, audio):
        x = self.encoder(audio).permute(0, 2, 1)
        x = self.proj_to_z(x).permute(0, 2, 1)
        poseterior = DiagonalGaussianDistribution(x)
        return poseterior

    def decode(self, latent):
        latent = self.proj_to_decoder(latent).permute(0, 2, 1)
        return self.decoder(self.latent_upsampler(latent))

    def forward(self, audio):
        posterior = self.encode(audio)
        latent = posterior.sample().permute(0, 2, 1)  # (b, t, latent_channel)
        recon_wav = self.decode(latent)
        return recon_wav, posterior
    

def chunk_latent(lat_pred: torch.Tensor, max_T: int):
    assert lat_pred.dim() == 3, "lat_pred 必须是 (B, T, C)"
    B, T, C = lat_pred.shape

    if T <= max_T:
        lat_chunks = lat_pred  # (B, T, C)
        N = 1
        T_chunk = T
        last_lat_len = T
        return lat_chunks, B, N, T_chunk, last_lat_len

    T_chunk = max_T
    N = math.ceil(T / T_chunk)
    pad_T = N * T_chunk - T   

    if pad_T > 0:
        pad = lat_pred.new_zeros(B, pad_T, C)
        lat_padded = torch.cat([lat_pred, pad], dim=1) 
    else:
        lat_padded = lat_pred

    # (B, N*T_chunk, C) -> (B, N, T_chunk, C) -> (B*N, T_chunk, C)
    lat_4d = lat_padded.view(B, N, T_chunk, C)
    lat_chunks = lat_4d.reshape(B * N, T_chunk, C)

    last_lat_len = T - (N - 1) * T_chunk

    return lat_chunks, B, N, T_chunk, last_lat_len


def unchunk_output(
    y_chunks: torch.Tensor,
    B: int,
    N: int,
    T_chunk: int,
    last_lat_len: int,
    upsample_ratio: float = 1.0,
    time_dim: int = 1,
):
    assert y_chunks.dim() >= 2, "输出至少要有 batch 和时间两个维度"
    Bn = y_chunks.shape[0]
    assert Bn == B * N, f"y_chunks 第一维={Bn}, 但 B*N={B*N}"

    if time_dim != 1:
        perm = list(range(y_chunks.dim()))
        perm[1], perm[time_dim] = perm[time_dim], perm[1]
        y_chunks = y_chunks.permute(*perm)

    Bn, T_out_chunk, *rest = y_chunks.shape
    y_chunks = y_chunks.view(B, N, T_out_chunk, *rest)  # (B, N, T_out_chunk, ...)

    valid_last_out_len = int(last_lat_len * upsample_ratio)
    valid_last_out_len = min(valid_last_out_len, T_out_chunk)

    T_full = (N - 1) * T_out_chunk + valid_last_out_len

    outputs = []
    for b in range(B):
        per_chunks = []
        for i in range(N):
            if i < N - 1:
                chunk = y_chunks[b, i, :T_out_chunk]
            else:
                chunk = y_chunks[b, i, :valid_last_out_len]
            per_chunks.append(chunk)
        full = torch.cat(per_chunks, dim=0)
        outputs.append(full)

    y_full = torch.stack(outputs, dim=0)  # (B, T_full, ...)

    if time_dim != 1:
        perm_back = list(range(y_full.dim()))
        perm_back[1], perm_back[time_dim] = perm_back[time_dim], perm_back[1]
        y_full = y_full.permute(*perm_back)

    return y_full


def vae_decode(vae, lat, max_tokens=1600, max_T=3000, upsample_ratio=960):
    bsz, T, _ = lat.shape
    
    if bsz == 1:
        return vae_decode(vae, lat, max_tokens=max_T, max_T=max_T, upsample_ratio=upsample_ratio)
    
    if T > max_tokens:
        
        wavs = []
        
        for i in range(bsz):
            
            lat_chunks, B, N, T_chunk, last_lat_len = chunk_latent(lat[i:i+1], max_tokens)  # (B*N, T_chunk, C)

            wav_chunks = vae.decode(lat_chunks)  # (B*N, 1, T_audio_chunk)
            wav_chunks = wav_chunks[:, 0]        # -> (B*N, T_audio_chunk)

            # 3) 拼接并去掉 padding 对应的尾部
            wav_full = unchunk_output(
                wav_chunks,
                B=B,
                N=N,
                T_chunk=T_chunk,
                last_lat_len=last_lat_len,
                upsample_ratio=upsample_ratio,
                time_dim=1,
            )
            
            wavs.append(wav_full)
        
        wav = torch.stack(wavs, dim=0)
        
        return wav

    else:
        
        wavs = []
        
        chunk_size = max_tokens // T
        n_chunks = math.ceil(bsz / chunk_size)
        for i in range(n_chunks):
            wavs_ = vae.decode(lat[i * chunk_size: (i + 1) * chunk_size])
            wavs.append(wavs_)
        
        wav = torch.cat(wavs, dim=0)

    return wav


if __name__ == '__main__':
    from utils.commons.hparams import hparams, set_hparams
    from utils.nn.model_utils import print_arch, num_params
    hp_vae = set_hparams('checkpoints/251120_wavvae_v4_unfreeze/config.yaml', global_hparams=False)
    vae = WavVAE_V3(hparams=hp_vae)
    
    print_arch(vae)
    num_params(vae)
    for n_, m_ in vae.named_children():
        num_params(m_, model_name=n_)
        
        
    # vae.eval()
    # vae.to('cuda')
    # vae.to(torch.bfloat16)
    
    # lat_pred = torch.rand(([8, 300, 32]), device='cuda', dtype=torch.bfloat16)
    
    # with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
    #     # wav_pred = vae.decode(lat_pred)[:, 0]
    #     wav_pred = vae_decode(vae, lat_pred)
    
    # print(f"{wav_pred = }")
    # print(f"{wav_pred.shape = }")
