import torch
import torch.nn as nn
from .lvcnet import LVCBlock
from ..commons.source import SourceModuleHnNSF, SourceModuleCycNoise_v1

MAX_WAV_VALUE = 32768.0


class Generator(nn.Module):
    """UnivNet Generator"""

    def __init__(self, h):
        super(Generator, self).__init__()
        self.h = h
        self.mel_channel = h['audio_num_mel_bins']
        self.noise_dim = h['gen']['noise_dim']
        self.hop_length = h['hop_size']
        channel_size = h['gen']['channel_size']
        kpnet_conv_size = h['gen']['kpnet_conv_size']

        if h['nsf_type'] != 'none':
            self.f0_upsamp = torch.nn.Upsample(scale_factor=self.hop_length)
            self.noise_convs = nn.ModuleList()
            if h['nsf_type'] == 'hn':
                self.harmonic_num = 7
                self.m_source = SourceModuleHnNSF(sampling_rate=h['audio_sample_rate'], harmonic_num=self.harmonic_num)
            if h['nsf_type'] == 'cycle':
                self.beta = 0.870
                self.m_source = SourceModuleCycNoise_v1(sampling_rate=h['audio_sample_rate'])
            c_out = 1
            for i, u in enumerate(h['gen']['strides'][::-1]):
                c_in = c_out
                c_out = channel_size
                self.noise_convs.append(nn.Conv1d(c_in, c_out, u * 2, u, padding=u // 2))

        self.res_stack = nn.ModuleList()
        hop_length = 1
        for stride in h['gen']['strides']:
            hop_length = stride * hop_length
            self.res_stack.append(
                LVCBlock(
                    channel_size,
                    h['audio_num_mel_bins'],
                    stride=stride,
                    dilations=h['gen']['dilations'],
                    lReLU_slope=h['gen']['lReLU_slope'],
                    cond_hop_length=hop_length,
                    kpnet_conv_size=kpnet_conv_size
                )
            )

        self.conv_pre = \
            nn.utils.weight_norm(nn.Conv1d(h['gen']['noise_dim'], channel_size, 7, padding=3, padding_mode='reflect'))

        self.conv_post = nn.Sequential(
            nn.LeakyReLU(h['gen']['lReLU_slope']),
            nn.utils.weight_norm(nn.Conv1d(channel_size, 1, 7, padding=3, padding_mode='reflect')),
            nn.Tanh(),
        )

    def forward(self, c, z, f0=None):
        '''
        Args: 
            c (Tensor): the conditioning sequence of mel-spectrogram (batch, mel_channels, in_length) 
            z (Tensor): the noise sequence (batch, noise_dim, in_length)
        
        '''
        if self.h['nsf_type'] != 'none':
            # harmonic-source signal, noise-source signal, uv flag
            f0 = self.f0_upsamp(f0[:, None]).transpose(1, 2)
            if self.h['nsf_type'] == 'cycle':
                beta = torch.ones(1, 1, 1, device=f0.device) * self.beta
                x_source, noi_source, uv = self.m_source(f0, beta)
            elif self.h['nsf_type'] == 'hn':
                x_source, noi_source, uv = self.m_source(f0)
            x_source = x_source.transpose(1, 2)
            x_sources = []
            for l in self.noise_convs:
                x_source = l(x_source)
                x_sources.append(x_source)

        z = self.conv_pre(z)  # (B, c_g, L)

        for i, res_block in enumerate(self.res_stack):
            res_block.to(z.device)
            if self.h['nsf_type'] != 'none':
                x_source = x_sources[len(self.res_stack) - i - 1]
                z = z + x_source
            z = res_block(z, c)  # (B, c_g, L * s_0 * ... * s_i)
        z = self.conv_post(z)  # (B, 1, L * 256)

        return z

    def eval(self, inference=False):
        super(Generator, self).eval()
        # don't remove weight norm while validation in training loop
        if inference:
            self.remove_weight_norm()

    def remove_weight_norm(self):
        print('Removing weight norm...')

        nn.utils.remove_weight_norm(self.conv_pre)

        for layer in self.conv_post:
            if len(layer.state_dict()) != 0:
                nn.utils.remove_weight_norm(layer)

        for res_block in self.res_stack:
            res_block.remove_weight_norm()

    def inference(self, c, z=None):
        # pad input mel with zeros to cut artifact
        # see https://github.com/seungwonpark/melgan/issues/8
        zero = torch.full((1, self.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)

        if z is None:
            z = torch.randn(1, self.noise_dim, mel.size(2)).to(mel.device)

        audio = self.forward(mel, z)
        audio = audio.squeeze()  # collapse all dimension except time axis
        audio = audio[:-(self.hop_length * 10)]
        audio = MAX_WAV_VALUE * audio
        audio = audio.clamp(min=-MAX_WAV_VALUE, max=MAX_WAV_VALUE - 1)
        audio = audio.short()

        return audio
