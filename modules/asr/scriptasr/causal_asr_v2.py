from dataclasses import dataclass
from typing import Any, Optional, Tuple
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
from modules.asr.mfa.nar_mfa_utils import build_alignment_from_durations, expected_index_one_hot

from utils.nn.seq_utils import add_prefix_nd, sequence_mask, remove_prefix, remove_suffix
from utils.nn.generation_utils import sample_topk, sample
from utils.commons.dataset_utils import pad_or_cut_xd

def build_asr_text_tokenizer():
    from utils.text.cosyvoice2_tokenizer import get_tokenizer
    text_tokenizer = get_tokenizer(multilingual=True, num_languages=100)
    vocab_size = text_tokenizer.encoding.n_vocab
    return text_tokenizer, vocab_size

def build_asr_model(hparams, text_tokenizer=None, init_pretrained=True, vocab_size=None, padding_idx=None):
    if text_tokenizer is not None:
        padding_idx = text_tokenizer.encode('<|endoftext|>')[0]
    model_config = ModelArgs(
        vocab_size=vocab_size,
        padding_idx=padding_idx,
        audio_encoder_type=hparams.get('audio_encoder_type'),
        audio_encoder_ckpt=hparams.get('audio_encoder_ckpt'),
        init_pretrained=init_pretrained,
    )
    from modules.asr.llama.llama_seq2seq import Seq2SeqLLaMA, ModelArgs as LLaMaS2SModelArgs
    model_config.backbone = 'llama_seq2seq'
    model_config.lm_config = LLaMaS2SModelArgs(
        enc_n_layers=4,
        dec_n_layers=20,
        use_gated_attention=True
    )
    model_config.lm_config.enc_n_layers = 4
    model_config.lm_config.dec_n_layers = 20
    if hparams.get('model_size', 'base') == 'small':
        model_config.lm_config.enc_n_layers = 2
        model_config.lm_config.dec_n_layers = 10
        model_config.lm_config.n_heads = 12
        model_config.lm_config.dim = 768 
    elif hparams.get('model_size', 'base') == '1b':
        model_config.lm_config.enc_n_layers = 8
        model_config.lm_config.dec_n_layers = 24
        model_config.lm_config.n_heads = 16
        model_config.lm_config.dim = 1536 

    model = CausalASRModel(model_config)
    return model


@dataclass
class ModelArgs:
    backbone = 'llama'
    lm_config = LLaMaModelArgs()

    vocab_size: int = None
    padding_idx: int = None

    audio_encoder_type: str = 'wavlm'
    audio_encoder_ckpt: str = None
    init_pretrained: bool = True

    aligner_n_layers: int = 2
    band_mask_width: int = 20

class CausalASRModel(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()

        from modules.asr.llama.llama_seq2seq import Seq2SeqLLaMA, ModelArgs as LLaMaS2SModelArgs
        self.lm_embed = nn.Embedding(config.vocab_size, config.lm_config.dim, config.padding_idx)
        self.lm = Seq2SeqLLaMA(config.lm_config)
        self.lm_head = nn.Linear(config.lm_config.dim, config.vocab_size, bias=False)

        from modules.asr.wavlm.WavLM import WavLM, WavLMConfig
        if config.init_pretrained:
            checkpoint = torch.load(config.audio_encoder_ckpt)
            cfg = WavLMConfig(checkpoint['cfg'])
            model = WavLM(cfg)
            model.load_state_dict(checkpoint['model'])
        else:
            cfg = json.load(open('checkpoints/wavlm/WavLM-Large-Config.json'))
            cfg = WavLMConfig(cfg)
            model = WavLM(cfg)
        self.audio_encoder_dim = cfg.encoder_embed_dim
        self.audio_encoder = model
        self.audio_encoder_hopsize = 320
        self.audio_encoder_sample_rate = 16000
        self.audio_proj = nn.Linear(self.audio_encoder_dim, config.lm_config.dim, bias=False)

        from modules.asr.llama.llama import LLaMa, ModelArgs as LLaMaModelArgs
        self.text_tower = LLaMa(LLaMaModelArgs(
            dim=config.lm_config.dim,
            n_layers=config.aligner_n_layers,
            n_heads=16,
            use_causal_attn=False,
            crossattn_n_layers=config.aligner_n_layers,
            use_qk_norm=True
        ))
        self.text_tower_proj = nn.Linear(config.lm_config.dim, config.lm_config.dim, bias=False)
        self.audio_tower = LLaMa(LLaMaModelArgs(
            dim=config.lm_config.dim,
            n_layers=config.aligner_n_layers,
            n_heads=16,
            use_causal_attn=False,
            crossattn_n_layers=config.aligner_n_layers,
            use_qk_norm=True
        ))
        self.audio_tower_proj = nn.Linear(config.lm_config.dim, config.lm_config.dim, bias=False)
        self.log_gamma = nn.Parameter(torch.log(torch.zeros(1)))  # [1]
    
        self.config = config

    def forward_audio_encoder(self, wavs, wav_mask, do_checkpoint=False):
        feat, feat_padding_mask = self.audio_encoder.extract_features(wavs, padding_mask=~(wav_mask.bool()), do_checkpoint=do_checkpoint)    # [B, T, C]
        feat_mask = ~feat_padding_mask  # [B, T]
        feat = self.audio_proj(feat)
        feat = pad_or_cut_xd(feat, tgt_len=wavs.shape[1] // self.audio_encoder_hopsize, dim=1)
        feat_mask = pad_or_cut_xd(feat_mask, tgt_len=wavs.shape[1] // self.audio_encoder_hopsize, dim=1)
        return feat, feat_mask
    
    def forward_txt_embed(self, txt_tokens):
        txt_embeds = self.lm_embed(txt_tokens)  # [B, T, C]
        return txt_embeds

    def forward_aligner(self, feat, feat_mask, txt_embeds, txt_mask, do_checkpoint=False):
        x_txt = self.text_tower(
            txt_embeds, txt_mask, context=feat, context_lens=feat_mask.sum(1), do_checkpoint=do_checkpoint
        )   # [B, T, C]
        x_txt = self.text_tower_proj(x_txt)
        feat = self.audio_tower(
            feat, feat_mask, context=txt_embeds, context_lens=txt_mask.sum(1), do_checkpoint=do_checkpoint
        )
        feat = self.audio_tower_proj(feat)

        alignment_logits = torch.bmm(feat, x_txt.transpose(1, 2)) / (feat.shape[2]**0.5)    # [B, T_mel, T_txt]
        alignment_mask = torch.bmm(feat_mask[..., None].to(x_txt), txt_mask[:, None, :].to(x_txt))

        # scale
        gamma = torch.exp(self.log_gamma).clamp(0.7, 3.0)[None, None]
        alignment_logits = alignment_logits * gamma.to(alignment_logits)

        min_val = torch.finfo(alignment_logits.dtype).min
        alignment_logits = alignment_logits.masked_fill(alignment_mask < 1, min_val)

        return alignment_logits, alignment_mask, gamma.detach()

        
    def forward(self, inputs, do_checkpoint=False):
        wavs = inputs['wavs']
        wav_mask = inputs['wav_mask']
        txt_tokens = inputs['txt_tokens_with_bot_eot']   # <BOT> ... <EOT>
        txt_mask = inputs['txt_mask_with_bot_eot']
        txt_lens = txt_mask.sum(1)
        bsz, T_txt = txt_tokens.shape

        feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask, do_checkpoint)
        feat_lens = feat_mask.sum(1)

        txt_embeds = self.forward_txt_embed(txt_tokens).to(feat)  # [B, T, C]

        x = self.lm(
            encoder_x=feat,
            encoder_padding_mask=feat_mask,
            decoder_x=txt_embeds,
            decoder_padding_mask=txt_mask,
            do_checkpoint=do_checkpoint
        )
        x = x[:, :-1]
        logits = self.lm_head(x)    # [B, T, C]
        labels = txt_tokens[:, 1:]  # [B, T]
        loss_mask = txt_mask[:, 1:]
        loss = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        loss = loss * loss_mask
        loss = loss.sum() / loss_mask.sum()

        outputs = {
            'logits': logits,
            'labels': labels,
            'ce_loss': loss,
            'ntokens': (feat_lens + txt_lens).sum()
        }

        ##################
        # alignment loss #
        ##################

        txt_tokens = inputs['txt_tokens']   # <BOT> ... <EOT>
        txt_mask = inputs['txt_mask']
        txt_embeds = self.forward_txt_embed(txt_tokens).to(feat)  # [B, T, C]

        seg_dur = inputs['seg_dur']
        seg_mask = inputs['seg_mask']   # [B, S]
        seg_align_pack = build_alignment_from_durations(seg_dur, seg_mask, ignore_index=0)
        frame_seg_idx = seg_align_pack['frame_labels']   # [B, T_mel] 每帧属于第几个 segment
        seg_mel_mask  = seg_align_pack['mel_mask']    # [B, T_mel]
        # 一般 seg_mel_mask 应该和上面的 mel_mask 一致/近似，可用你原来的 mel_mask

        device = txt_tokens.device
        token_seg_id = inputs['token_seg_id']
        # seg_token_mask[b, s, t] = True 表示第 b 条样本的第 s 段包含第 t 个 token
        seg_token_mask = torch.zeros(bsz, seg_dur.shape[1], T_txt, dtype=torch.bool, device=device)
        arange_s = torch.arange(seg_dur.shape[1], device=device)[None, :, None]      # [1, S, 1]
        token_seg_id_exp = token_seg_id[:, None, :]                    # [B, 1, T_txt]
        seg_token_mask = (token_seg_id_exp == arange_s) & txt_mask[:, None, :]
        # 再 AND 上 seg_mask 防一下 padding 段
        seg_token_mask = seg_token_mask & seg_mask[:, :, None]

        alignment_logits, alignment_mask, gamma = self.forward_aligner(feat, feat_mask, txt_embeds, txt_mask, do_checkpoint)

        log_p = F.log_softmax(alignment_logits, dim=-1)  # [B, T_mel, T_txt]

        B, T_mel, T_txt = alignment_logits.shape
        S = seg_dur.shape[1]
        device = alignment_logits.device

        # 1) 把 frame_seg_idx 转成 one-hot，方便和 seg_token_mask 做乘积
        frame_seg_idx_clamped = frame_seg_idx.clamp(min=0)      # 无效帧先当 0
        frame_seg_onehot = F.one_hot(frame_seg_idx_clamped, num_classes=S).float()  # [B, T_mel, S]

        # 对于无效帧（mel_mask = 0）强制 one-hot 为 0
        frame_seg_onehot = frame_seg_onehot * feat_mask[..., None].float()   # [B, T_mel, S]

        # 2) 对每个帧 t，计算这个帧所允许的 token mask：
        #    allowed_tokens[b, t, i] = 1  当且仅当 这个帧所属的段 s* 包含 token i
        #    本质上是：one-hot(frame_seg) * seg_token_mask 做一个 “按段聚合”
        # allowed_tokens = torch.einsum("bts,bst->bti", frame_seg_onehot, seg_token_mask.float()).bool()
        allowed_tokens = torch.einsum(
            "bts,bsi->bti",   # 注意这里：bts, bsi -> bti
            frame_seg_onehot,
            seg_token_mask.float()
        ).bool()
        # [B, T_mel, T_txt] bool, 每帧有哪些 token 是允许的（在对应 segment 内）

        # 3) mask 掉不在该段内的 token
        neg_inf = torch.finfo(log_p.dtype).min
        log_p_masked = torch.where(allowed_tokens, log_p, neg_inf)  # [B, T_mel, T_txt]

        # 4) 对 allowed token 做 logsumexp -> 对应公式中的 sum_{i in G(s*)} p_{t,i}
        log_prob_correct_seg = torch.logsumexp(log_p_masked, dim=-1)  # [B, T_mel]

        # 5) 对有效帧做 NLL
        alignment_loss_seg = -(log_prob_correct_seg * feat_mask.float()).sum() / feat_mask.sum()

        ##################
        # aggregate loss #
        ##################
        alignment_probs = torch.softmax(alignment_logits, dim=2)    # [B, T_mel, T_txt]
        dur_soft_tok = (alignment_probs * feat_mask[..., None]).sum(1)    # [B, T_txt]

        # token_seg_id: [B, T_txt]，上面构造过
        B, T_txt = dur_soft_tok.shape
        S = seg_dur.shape[1]
        device = dur_soft_tok.device
        dur_soft_seg = torch.zeros(B, S, device=device, dtype=dur_soft_tok.dtype)
        valid_tok = token_seg_id >= 0
        b_ids = torch.arange(B, device=device).unsqueeze(1).expand(B, T_txt)[valid_tok]
        s_ids = token_seg_id[valid_tok]
        dur_vals = dur_soft_tok[valid_tok]
        dur_soft_seg.index_put_((b_ids, s_ids), dur_vals, accumulate=True)

        dur_tgt_seg = seg_dur.to(dur_soft_seg)

        seg_mask_float = seg_mask.float()
        dur_pred_loss = ((dur_soft_seg - dur_tgt_seg).abs() * seg_mask_float).sum() / seg_mask_float.sum()
        dur_pred_loss_log = ((torch.log1p(dur_soft_seg.clamp_min(0)) - torch.log1p(dur_tgt_seg.clamp_min(0))).pow(2) * seg_mask_float).sum() / seg_mask_float.sum()
        dur_pred_loss = dur_pred_loss + dur_pred_loss_log

        #############
        # mono loss #
        #############
        expected_idx = (alignment_probs * torch.arange(txt_tokens.shape[1]).to(alignment_probs)[None, None, :]).sum(dim=-1) # [B, T_mel]
        mono_loss = F.relu(expected_idx[:, :-1] - expected_idx[:, 1:])  # [B, T_mel - 1]
        mono_mask = sequence_mask((feat_mask.sum(1) - 1).clamp_min(0), maxlen=mono_loss.shape[1])
        mono_loss = (mono_loss * mono_mask).sum() / mono_mask.sum()

        outputs.update({
            'align_loss': alignment_loss_seg,
            'bd_agg_loss': dur_pred_loss,
            'mono_loss': mono_loss,
            'gamma': gamma
        })

        return outputs

    @torch.no_grad()
    def inference(self, wavs, txt_tokens, 
                  topk=1, temperature=0.1, max_new_tokens=512, eos_idx=None, 
                  diarization=False,
                  use_tqdm=True, print_candidates=False):
        wav_mask = torch.ones_like(wavs).int()
        feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask)

        txt_mask = torch.ones_like(txt_tokens)
        txt_embeds = self.forward_txt_embed(txt_tokens).to(feat)  # [B, T, C]

        feat_lens = feat_mask.sum(1)
        txt_lens = txt_mask.sum(1)
        if self.config.backbone != 'llama_seq2seq':
            x = add_prefix_nd(feat, feat_lens, txt_embeds, txt_lens)

        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(max_new_tokens), desc='| Generating')
        else:
            it = range(max_new_tokens)


        if self.config.backbone == 'llama':

            x_ = self.lm(x, attn_mask=sequence_mask(feat_lens + txt_lens), start_pos=0, use_cache=True)

            def forward_(token, start_pos):
                x = self.lm_embed(token)
                x = self.lm(x, attn_mask=None, start_pos=start_pos, use_cache=True)
                logits = self.lm_head(x)
                return logits

            start_pos = (feat_lens + txt_lens - 1).to(torch.int32)
            for step in it:
                logits = forward_(txt_tokens[:, -1:], start_pos)
                token_pred = self.sample(logits, topk, temperature, print_candidates)
                if token_pred[0] == eos_idx:
                    break
                txt_tokens = torch.cat([txt_tokens, token_pred], dim=1)
                start_pos = start_pos + 1

        elif self.config.backbone == 'llama_seq2seq':

            self.lm.reset_kv_cache()
            start_pos = 0

            enc_out = self.lm.encode(
                encoder_x=feat,
                encoder_padding_mask=feat_mask,
            )

            def forward_(token, start_pos):
                x = self.lm_embed(token)
                x = self.lm.decode(
                    decoder_x=x,
                    decoder_padding_mask=None,
                    enc_out=enc_out,
                    encoder_padding_mask=feat_mask,
                    start_pos=start_pos,
                    use_cache=True
                )
                logits = self.lm_head(x)
                return logits

            for step in it:
                logits = forward_(txt_tokens[:, -1:], start_pos)
                token_pred = self.sample(logits, topk, temperature, print_candidates)
                if token_pred[0] == eos_idx:
                    break
                txt_tokens = torch.cat([txt_tokens, token_pred], dim=1)
                start_pos = start_pos + 1

        elif self.config.backbone == 'qwen3':

            past_key_values = None
            for step in it:
                if step == 0:
                    input_ids = None
                    inputs_embeds = x
                    logits_to_keep = 0
                else:
                    input_ids = txt_tokens[:, -1:]
                    inputs_embeds = None
                    logits_to_keep = 1
                lm_outputs = self.lm(
                    input_ids=input_ids, inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    use_cache=True, logits_to_keep=logits_to_keep,
                    output_hidden_states=True
                )
                if step == 0:
                    x_ = lm_outputs.hidden_states[-1]
                logits = lm_outputs.logits
                past_key_values = lm_outputs.past_key_values
                token_pred = self.sample(logits, topk, temperature, print_candidates)
                if token_pred[0] == eos_idx:
                    break
                txt_tokens = torch.cat([txt_tokens, token_pred], dim=1)
        
        if not diarization:
            return txt_tokens[:, txt_lens[0]:]
        
        if self.config.model_spk_diarization:
            if self.config.spk_diarization_after_lm:
                feat_ = remove_suffix(x_, feat_lens)
                spk_mask_logits = self.forward_spk_decoder(feat_, feat_mask)
            else:
                spk_mask_logits = self.forward_spk_decoder(feat, feat_mask)
            spk_mask_logits = torch.sigmoid(spk_mask_logits)

        return txt_tokens[:, txt_lens[0]:], spk_mask_logits


    def sample(self, logits, topk, temperature, print_candidates=False):
        if topk == 1:
            token_pred = torch.argmax(logits[:, -1:], dim=-1)   # [1, 1]
        else:
            # token_pred = sample_topk(logits, topk, temperature)[0, 0]
            token_pred = sample(logits[:, -1:], topk, 1.0, temperature)
        if print_candidates and hasattr(self, 'text_tokenizer'):
            topk_logits, topk_idxs = torch.topk(torch.softmax(logits[:, -1], dim=1), 5)
            candidates = []
            for topk_logit, topk_idx in zip(topk_logits[0].cpu().numpy().tolist(), topk_idxs[0].cpu().numpy().tolist()):
                candidates.append({self.text_tokenizer.decode([topk_idx]): round(topk_logit, 3)})
            print(candidates, end=' ')
            print(f'Chosen: {self.text_tokenizer.decode([token_pred[-1].item()])}')

        return token_pred
    
    @torch.no_grad()
    def inference_batch(self, wavs, wav_mask, txt_tokens, txt_mask, topk=1, temperature=0.1, 
                        max_new_tokens=512, eos_idx=None, use_tqdm=True):
        bsz = len(wavs)
        
        if use_tqdm:
            from tqdm import tqdm
            it = tqdm(range(max_new_tokens), desc='| Generating')
        else:
            it = range(max_new_tokens)
            
        if self.config.backbone == 'llama':
            # TODO
            feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask)
            
            feat_lens = feat_mask.sum(1)
            txt_lens = txt_mask.sum(1)
            
        elif self.config.backbone == 'llama_seq2seq':
            feat, feat_mask = self.forward_audio_encoder(wavs, wav_mask)
            
            feat_lens = feat_mask.sum(1)
            txt_lens = txt_mask.sum(1)
            
            self.lm.reset_kv_cache()
            start_pos = 0
            
            enc_out = self.lm.encode(
                encoder_x=feat,
                encoder_padding_mask=feat_mask,
            )
            
            def forward_(token, start_pos):
                x = self.lm_embed(token)
                x = self.lm.decode(
                    decoder_x=x,
                    decoder_padding_mask=None,
                    enc_out=enc_out,
                    encoder_padding_mask=feat_mask,
                    start_pos=start_pos,
                    use_cache=True
                )
                logits = self.lm_head(x)
                return logits
            
            stop = torch.zeros(bsz).to(txt_tokens)
            for step in it:
                logits = forward_(txt_tokens[:, -1:], start_pos)
                tokens_pred = self.sample(logits, topk, temperature)
                stop[tokens_pred[..., 0] == eos_idx] = 1
                if stop.sum() == bsz:
                    break
                txt_tokens = torch.cat([txt_tokens, tokens_pred], dim=1)
                start_pos = start_pos + 1

        results = []
        for i in range(bsz):
            tokens_pred = txt_tokens[i, txt_lens[i]:]
            end_pos = torch.nonzero(tokens_pred == eos_idx)
            if len(end_pos) >= 1:
                tokens_pred = tokens_pred[:end_pos[0, 0]]
            # print(f"{len(end_pos) = }")
            # print(f"{end_pos = }")
            # print(f"{tokens_pred = }")
            results.append(tokens_pred)
            
        return results
