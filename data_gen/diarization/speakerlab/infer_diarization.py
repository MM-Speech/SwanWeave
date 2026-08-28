# Copyright 3D-Speaker (https://github.com/alibaba-damo-academy/3D-Speaker). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

"""
This is a speaker diarization inference script based on pretrained models.
Usages:
    1. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir]
    2. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token]
    3. python infer_diarization.py --wav [wav_list OR wav_path] --out_dir [out_dir] --include_overlap --hf_access_token [hf_access_token] --nprocs [n]
"""

import os
import sys
import argparse
import warnings
from typing import Any, Dict

import numpy as np
from tqdm import tqdm
import json

import torch
import torch.multiprocessing as mp

try:
    from speakerlab.utils.config import Config
except ImportError:
    sys.path.append('%s/../..'%os.path.dirname(os.path.abspath(__file__)))
    sys.path.append('%s/..'%os.path.dirname(os.path.abspath(__file__)))
    from speakerlab.utils.config import Config

from speakerlab.utils.builder import build
from speakerlab.utils.utils import merge_vad, silent_print, download_model_from_modelscope, circle_pad
from speakerlab.utils.fileio import load_audio

warnings.filterwarnings("ignore")

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

parser = argparse.ArgumentParser(description='Speaker diarization inference.')
parser.add_argument('--wav', type=str, required=True, help='Input wavs')
parser.add_argument('--out_dir', type=str, required=True, help='Out results dir')
parser.add_argument('--out_type', choices=['rttm', 'json'], default='rttm', type=str, help='Results format, rttm or json')
parser.add_argument('--include_overlap', action='store_true', help='Include overlapping region')
parser.add_argument('--hf_access_token', type=str, help='hf_access_token for pyannote/segmentation-3.0 model. It\'s required if --include_overlap is specified')
parser.add_argument('--diable_progress_bar', action='store_true', help='Close the progress bar')
parser.add_argument('--nprocs', default=None, type=int, help='Num of procs')
parser.add_argument('--speaker_num', default=None, type=int, help='Oracle num of speaker')


def get_speaker_embedding_model(device:torch.device = None, cache_dir:str = None):
    conf = {
        'model_id': 'iic/speech_campplus_sv_zh_en_16k-common_advanced',
        'revision': 'v1.0.0',
        'model_ckpt': 'campplus_cn_en_common.pt',
        'embedding_model': {
            'obj': 'speakerlab.models.campplus.DTDNN.CAMPPlus',
            'args': {
                'feat_dim': 80,
                'embedding_size': 192,
            },
        },
        'feature_extractor': {
            'obj': 'speakerlab.process.processor.FBank',
            'args': {
                'n_mels': 80,
                'sample_rate': 16000,
                'mean_nor': True,
                },
        }
    }

    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    # cache_dir = '/mnt/bn/sa-ag-data/liruiqi/code/modelscope/models/iic/speech_campplus_sv_zh_en_16k-common_advanced'
    pretrained_model_path = os.path.join(cache_dir, conf['model_ckpt'])
    config = Config(conf)
    feature_extractor = build('feature_extractor', config)
    embedding_model = build('embedding_model', config)

    # load pretrained model
    pretrained_state = torch.load(pretrained_model_path, map_location='cpu')
    embedding_model.load_state_dict(pretrained_state)
    embedding_model.eval()
    if device is not None:
        embedding_model.to(device)
    return embedding_model,  feature_extractor

def get_cluster_backend(
    cluster_type='spectral',
    mer_cos=0.8,
    min_num_spks=1,
    max_num_spks=15,
    min_cluster_size=4,
    oracle_num=None,
    pval=0.012,
    cluster_line=40,
):
    conf = {
        'cluster': {
            'obj': 'speakerlab.process.cluster.CommonClustering',
            'args': {
                'cluster_type': cluster_type,
                'cluster_line': cluster_line,
                'mer_cos': mer_cos,
                'min_num_spks': min_num_spks,
                'max_num_spks': max_num_spks,
                'min_cluster_size': min_cluster_size,
                'oracle_num': oracle_num,
                'pval': pval,
            }
        }
    }
    config = Config(conf)
    return build('cluster', config)

def _modelscope_device_str(device: torch.device | None) -> str:
    if device is None:
        return 'cpu'
    if device.type == 'cpu':
        return 'cpu'
    if device.type == 'cuda':
        # torch.device('cuda') => index is None
        # torch.device('cuda:0') => index is 0 (falsy), so must check against None
        return device.type if device.index is None else f'{device.type}:{device.index}'
    return str(device)


def get_voice_activity_detection_model(device: torch.device=None, cache_dir:str = None):
    conf = {
        'model_id': 'iic/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        'revision': 'v2.0.4',
    }
    cache_dir = download_model_from_modelscope(conf['model_id'], conf['revision'], cache_dir)
    # cache_dir = '/mnt/bn/sa-ag-data/liruiqi/code/modelscope/models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch'
    with silent_print():
        vad_pipeline = pipeline(
            task=Tasks.voice_activity_detection,
            model=cache_dir,
            device=_modelscope_device_str(device),
            disable_pbar=True,
            disable_update=True,
        )
    return vad_pipeline

def get_segmentation_model(use_auth_token, device: torch.device=None):
    # Lazy import: avoid importing pyannote.audio unless overlap handling is requested.
    try:
        from pyannote.audio import Inference, Model
    except Exception as e:
        raise ImportError(
            'pyannote.audio is required only when include_overlap=True. '
            'Please install pyannote.audio and its dependencies.'
        ) from e

    segmentation_params = {
        'segmentation': 'pyannote/segmentation-3.0',
        'segmentation_batch_size': 32,
        'use_auth_token': use_auth_token,
    }
    model = Model.from_pretrained(
        segmentation_params['segmentation'],
        use_auth_token=segmentation_params['use_auth_token'],
        strict=False,
    )
    segmentation = Inference(
        model,
        duration=model.specifications.duration,
        step=0.1 * model.specifications.duration,
        skip_aggregation=True,
        batch_size=segmentation_params['segmentation_batch_size'],
        device=device,
    )
    return segmentation


class Diarization3Dspeaker():
    """
    This class is designed to handle the speaker diarization process, 
    which involves identifying and segmenting audio by speaker identities. 
    Args:
        device (str, default=None): The device on which models will run. 
        include_overlap (bool, default=False): Indicates whether to include overlapping 
            speech segments in the diarization output. Overlapping speech occurs when multiple 
            speakers are talking simultaneously.
        hf_access_token (str, default=None): Access token for Hugging Face, required if 
            include_overlap is True. This token allows access to pynnote segmentation models 
            available on the Hugging Face that handles overlapping speech.
        speaker_num (int, default=None): Specify number of speakers.
        model_cache_dir (str, default=None): If specified, the pretrained model will be downloaded 
            to this directory; only pretrained from modelscope are supported.
    Usage:
        diarization_pipeline = Diarization3Dspeaker(device, include_overlap, hf_access_token)
        output = diarization_pipeline(input_audio) # input_audio can be a path to a WAV file, a NumPy array, or a PyTorch tensor
        print(output) # output: [[1.1, 2.2, 0], [3.1, 4.1, 1], ..., [st_n, ed_n, speaker_id]]
        diarization_pipeline.save_diar_output('audio.rttm') # or audio.json
    """
    def __init__(self, device=None, include_overlap=False, hf_access_token=None, speaker_num=None, model_cache_dir=None):
        if include_overlap and hf_access_token is None:
            raise ValueError("hf_access_token is required when include_overlap is True.")

        self.device = self.normalize_device(device)
        self.include_overlap = include_overlap

        self.embedding_model, self.feature_extractor = get_speaker_embedding_model(self.device, model_cache_dir)
        self.vad_model = get_voice_activity_detection_model(self.device, model_cache_dir)
        self.cluster = get_cluster_backend()
        self.count_cluster = get_cluster_backend(
            cluster_type='spectral',
            cluster_line=1,
            mer_cos=None,
            min_num_spks=1,
            max_num_spks=8,
            min_cluster_size=0,
            oracle_num=None,
            pval=0.02,
        )
        self.count_mode_chunk_dur = 0.75
        self.count_mode_chunk_step = 0.25

        if include_overlap:
            self.segmentation_model = get_segmentation_model(hf_access_token, self.device)
        
        self.batchsize = 64
        self.fs = self.feature_extractor.sample_rate
        self.output_field_labels = None
        self.speaker_num = speaker_num

    def __call__(self, wav, wav_fs=None, speaker_num=None):
        wav_data = load_audio(wav, wav_fs, self.fs)

        # stage 1-1: do vad
        vad_time = self.do_vad(wav_data)
        if self.include_overlap:
            # stage 1-2: do segmentation
            segmentations, count = self.do_segmentation(wav_data)
            valid_field = get_valid_field(count)
            vad_time = merge_vad(vad_time, valid_field)

        # stage 2: prepare subseg
        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed)]

        # stage 3: extract embeddings
        embeddings = self.do_emb_extraction(chunks, wav_data)

        # stage 4: clustering
        speaker_num, output_field_labels = self.do_clustering(chunks, embeddings, speaker_num)

        if self.include_overlap:
            # stage 5: include overlap results
            binary = self.post_process(output_field_labels, speaker_num, segmentations, count)
            timestamps = [count.sliding_window[i].middle for i in range(binary.shape[0])]
            output_field_labels = self.binary_to_segs(binary, timestamps)

        self.output_field_labels = output_field_labels
        return output_field_labels

    def analyze_speakers(
        self,
        wav,
        wav_fs=None,
        include_segments: bool = True,
        include_embedding_diagnostics: bool = True,
        return_embeddings: bool = False,
        count_mode: bool = False,
        count_mode_chunk_dur: float | None = None,
        count_mode_chunk_step: float | None = None,
        outlier_std_scale: float = 2.0,
        outlier_abs_threshold: float | None = None,
        outlier_min_count: int = 2,
        outlier_min_ratio: float = 0.05,
        outlier_max_windows: int = 20,
        early_stop: bool = False,
        early_stop_min_chunks: int = 64,
        early_stop_on_outlier: bool = True,
        early_stop_on_multispeaker: bool = True,
        early_stop_check_every: int = 96,
    ) -> Dict[str, Any]:
        """Analyze speaker count with richer diagnostics.

        This method does NOT change the existing ``__call__`` behavior. It reuses the
        same core pipeline (VAD -> chunking -> embedding -> clustering), but returns a
        structured analysis result instead of only diarization segments.

        About ``count_mode``:
            - ``count_mode=False`` uses the default diarization-oriented setup.
            - ``count_mode=True`` uses a more sensitive speaker-counting setup, such as
              shorter chunks and a less aggressive clustering configuration, so that
              short and weak secondary speakers are less likely to be merged away.
            - ``count_mode`` is NOT a "single-speaker mode". The method still performs
              normal clustering and can return ``num_speakers`` as 1, 2, 3, ... based
              on the observed audio.

        About ``outlier_evidence``:
            - ``outlier_evidence`` is designed to catch weak secondary-speaker evidence
              when clustering still collapses the whole audio into a single cluster.
            - Therefore, the current ``outlier_evidence.detected=True`` logic is only
              promoted when ``num_speakers == 1``. In that case it means: the main
              clustering result is still single-speaker, but there are enough windows
              that deviate strongly from the main speaker centroid, so a weak secondary
              speaker may have been missed.
            - If ``num_speakers > 1``, the method still returns outlier statistics, but
              it does NOT currently upgrade them into a "possible extra speaker"
              conclusion. In other words, ``detected=False`` under a multi-speaker
              result does NOT mean there are no additional speakers beyond the current
              clustered speaker count.

        Important notes and limitations:
            - ``num_speakers`` should be interpreted as the number of stable clusters
              found by the current pipeline, not as a guaranteed ground-truth speaker
              count. In complex audio it may under-estimate the real number of speakers
              if weak / short / overlapping speakers are merged into stronger ones.
            - ``confidence`` is a heuristic stability score for the current partition,
              not a calibrated probability that the estimated speaker count is correct.
              In ``count_mode=True`` it is best read together with
              ``num_speakers``, ``warnings`` and ``outlier_evidence``.
            - ``count_mode=True`` improves sensitivity to weak or short speakers, but
              this usually increases the risk of over-segmentation and false positives.
            - Chunk-level durations in ``chunks.label_durations`` and
              ``chunks.total_chunk_duration`` are overlap-counted statistics because
              neighboring chunks overlap in time; they are not equal to the true audio
              duration covered by that speaker. For more human-readable durations,
              prefer ``segments.speaker_durations``.

        Args:
            wav: Input audio. Can be a wav path, NumPy array, or PyTorch tensor.
            wav_fs: Original sampling rate when ``wav`` is an array/tensor.
            include_segments: Whether to return segment-level statistics.
            include_embedding_diagnostics: Whether to compute embedding-based metrics
                such as intra-cluster similarity and centroid similarity.
            return_embeddings: Whether to include raw embeddings in the result.
            count_mode: Whether to use the more sensitive speaker-counting setup.
            count_mode_chunk_dur: Optional chunk duration override used only when
                ``count_mode=True``.
            count_mode_chunk_step: Optional chunk step override used only when
                ``count_mode=True``.
            outlier_std_scale: Dynamic outlier threshold scale. A smaller value makes
                outlier detection more sensitive.
            outlier_abs_threshold: Optional absolute lower bound for outlier detection.
            outlier_min_count: Minimum number of outlier windows required before
                ``outlier_evidence.detected`` can become True.
            outlier_min_ratio: Minimum outlier-window ratio required before
                ``outlier_evidence.detected`` can become True.
            outlier_max_windows: Maximum number of outlier windows to keep in the
                returned result payload.

        Returns:
            dict with keys such as:
                - ``num_speakers``: clustering-estimated stable speaker-cluster count
                - ``confidence``: heuristic partition-stability score in [0, 1]
                - ``confidence_breakdown``: detailed confidence diagnostics
                - ``outlier_evidence``: weak-secondary-speaker evidence summary
                - ``warnings``: warning flags
                - ``speech``: VAD/audio-duration statistics
                - ``chunks``: chunk-level labels and distributions
                - ``segments``: optional segment-level summary
        """
        wav_data = load_audio(wav, wav_fs, self.fs)

        audio_duration = float(wav_data.shape[-1]) / float(self.fs) if hasattr(wav_data, 'shape') else 0.0
        vad_time = self.do_vad(wav_data)
        speech_duration = float(sum(max(0.0, ed - st) for st, ed in vad_time))
        speech_ratio = (speech_duration / audio_duration) if audio_duration > 0 else 0.0

        warnings_list = []
        if speech_duration <= 0.0 or len(vad_time) == 0:
            warnings_list.append('no_speech_detected')
            return {
                'num_speakers': 0,
                'confidence': 0.0,
                'confidence_breakdown': {
                    'reason': 'no_speech_detected',
                },
                'warnings': warnings_list,
                'speech': {
                    'vad_intervals': vad_time,
                    'audio_duration': audio_duration,
                    'speech_duration': speech_duration,
                    'speech_ratio': speech_ratio,
                },
                'chunks': {
                    'times': [],
                    'labels': [],
                    'label_ids_original': [],
                    'label_counts': {},
                    'label_ratios': {},
                    'label_durations': {},
                },
                'segments': {
                    'segs': [],
                    'speaker_durations': {},
                    'speaker_turns': {},
                } if include_segments else None,
            }

        # stage 2: prepare subseg
        chunk_dur = (count_mode_chunk_dur if count_mode_chunk_dur is not None else self.count_mode_chunk_dur) if count_mode else 1.5
        chunk_step = (count_mode_chunk_step if count_mode_chunk_step is not None else self.count_mode_chunk_step) if count_mode else 0.75
        chunks = [c for (st, ed) in vad_time for c in self.chunk(st, ed, dur=chunk_dur, step=chunk_step)]
        if len(chunks) == 0:
            warnings_list.append('no_chunks_generated')
            return {
                'num_speakers': 0,
                'confidence': 0.0,
                'confidence_breakdown': {
                    'reason': 'no_chunks_generated',
                },
                'warnings': warnings_list,
                'speech': {
                    'vad_intervals': vad_time,
                    'audio_duration': audio_duration,
                    'speech_duration': speech_duration,
                    'speech_ratio': speech_ratio,
                },
                'chunks': {
                    'times': [],
                    'labels': [],
                    'label_ids_original': [],
                    'label_counts': {},
                    'label_ratios': {},
                    'label_durations': {},
                },
                'segments': {
                    'segs': [],
                    'speaker_durations': {},
                    'speaker_turns': {},
                } if include_segments else None,
            }

        # stage 3: extract embeddings (batched). Optionally early-stop for faster rejection.
        cluster_backend = self.count_cluster if count_mode else self.cluster
        total_num_chunks = len(chunks)
        embeddings_list = []

        # Online stats for early-stop outlier detection (single-speaker heuristic).
        online_n = 0
        online_mean = 0.0
        online_m2 = 0.0
        online_centroid = None  # normalized centroid
        early_outlier_indices: list[int] = []
        early_outlier_scores: list[float] = []
        early_dynamic_threshold = None
        early_threshold = None

        def _online_update(x: float):
            nonlocal online_n, online_mean, online_m2
            online_n += 1
            delta = x - online_mean
            online_mean += delta / online_n
            delta2 = x - online_mean
            online_m2 += delta * delta2

        batch_st = 0
        with torch.no_grad():
            while batch_st < total_num_chunks:
                batch_chunks = chunks[batch_st: batch_st + self.batchsize]
                wavs = [
                    wav_data[0, int(st * self.fs):int(ed * self.fs)]
                    for st, ed in batch_chunks
                ]
                max_len = max([x.shape[0] for x in wavs]) if len(wavs) > 0 else 0
                wavs = [circle_pad(x, max_len) for x in wavs]
                wavs = torch.stack(wavs).unsqueeze(1)

                wavs_batch = wavs.to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                emb_batch = self.embedding_model(feats_batch).detach().cpu().numpy()
                embeddings_list.append(emb_batch)

                if early_stop:
                    embn = emb_batch.astype(np.float32)
                    embn = embn / (np.linalg.norm(embn, axis=1, keepdims=True) + 1e-12)

                    for i in range(embn.shape[0]):
                        gidx = batch_st + i
                        e = embn[i]
                        if online_centroid is None:
                            online_centroid = e.copy()
                        sim = float(np.dot(e, online_centroid))
                        _online_update(sim)

                        # update centroid as running mean and re-normalize
                        online_centroid = online_centroid + (e - online_centroid) / float(online_n)
                        online_centroid = online_centroid / (np.linalg.norm(online_centroid) + 1e-12)

                        if online_n >= max(8, outlier_min_count * 2):
                            std = float(np.sqrt(online_m2 / max(1, online_n - 1)))
                            early_dynamic_threshold = online_mean - outlier_std_scale * std
                            early_threshold = early_dynamic_threshold if outlier_abs_threshold is None else max(
                                early_dynamic_threshold, outlier_abs_threshold
                            )
                            if sim < early_threshold:
                                early_outlier_indices.append(gidx)
                                early_outlier_scores.append(sim)

                    if early_stop_on_outlier and online_n >= early_stop_min_chunks:
                        outlier_ratio = float(len(early_outlier_indices) / float(online_n))
                        if len(early_outlier_indices) >= outlier_min_count and outlier_ratio >= outlier_min_ratio:
                            keep_idx = early_outlier_indices[:outlier_max_windows]
                            keep_scores = early_outlier_scores[:outlier_max_windows]
                            confidence_early = float(np.clip(online_mean, 0.0, 1.0) * min(1.0, online_n / 24.0))
                            return {
                                'num_speakers': 1,
                                'confidence': confidence_early,
                                'confidence_breakdown': {
                                    'count_mode': count_mode,
                                    'early_stopped': True,
                                    'early_stop_reason': 'outlier_detected',
                                    'processed_num_chunks': int(online_n),
                                    'total_num_chunks': int(total_num_chunks),
                                    'mean_intra_cos': float(online_mean),
                                    'std_intra_cos': float(np.sqrt(online_m2 / max(1, online_n - 1))),
                                },
                                'outlier_evidence': {
                                    'detected': True,
                                    'count': int(len(early_outlier_indices)),
                                    'ratio': outlier_ratio,
                                    'threshold': None if early_threshold is None else float(early_threshold),
                                    'dynamic_threshold': None if early_dynamic_threshold is None else float(early_dynamic_threshold),
                                    'indices': keep_idx,
                                    'times': chunk_times[np.array(keep_idx, dtype=np.int64)].tolist(),
                                    'scores': [float(s) for s in keep_scores],
                                    'params': {
                                        'outlier_std_scale': float(outlier_std_scale),
                                        'outlier_abs_threshold': outlier_abs_threshold,
                                        'outlier_min_count': int(outlier_min_count),
                                        'outlier_min_ratio': float(outlier_min_ratio),
                                        'outlier_max_windows': int(outlier_max_windows),
                                    },
                                },
                                'warnings': ['possible_minor_speaker', 'early_stopped'],
                                'speech': {
                                    'vad_intervals': vad_time,
                                    'audio_duration': audio_duration,
                                    'speech_duration': speech_duration,
                                    'speech_ratio': speech_ratio,
                                },
                                'chunks': {
                                    'times': chunk_times[:online_n].tolist(),
                                    'labels': [0] * int(online_n),
                                    'label_ids_original': [0],
                                    'label_counts': {'0': int(online_n)},
                                    'label_ratios': {'0': 1.0},
                                    'label_durations': {},
                                    'total_chunk_duration': float(np.sum(chunk_durs[:online_n])),
                                },
                                'segments': None if not include_segments else {
                                    'segs': [],
                                    'speaker_durations': {},
                                    'speaker_turns': {},
                                },
                            }

                    if early_stop_on_multispeaker and online_n >= early_stop_min_chunks:
                        # Periodically run clustering on the prefix. If it already yields >1 speakers,
                        # we can reject early for strict single-speaker filtering.
                        if (online_n - early_stop_min_chunks) % max(1, early_stop_check_every) == 0:
                            emb_prefix = np.concatenate(embeddings_list, axis=0)
                            labels_prefix = np.asarray(cluster_backend(emb_prefix, speaker_num=None), dtype=np.int64)
                            k_prefix = int(np.unique(labels_prefix).size)
                            if k_prefix > 1:
                                return {
                                    'num_speakers': k_prefix,
                                    'confidence': 0.0,
                                    'confidence_breakdown': {
                                        'count_mode': count_mode,
                                        'early_stopped': True,
                                        'early_stop_reason': 'multispeaker_detected',
                                        'processed_num_chunks': int(online_n),
                                        'total_num_chunks': int(total_num_chunks),
                                    },
                                    'outlier_evidence': None,
                                    'warnings': ['multispeaker_detected', 'early_stopped'],
                                    'speech': {
                                        'vad_intervals': vad_time,
                                        'audio_duration': audio_duration,
                                        'speech_duration': speech_duration,
                                        'speech_ratio': speech_ratio,
                                    },
                                    'chunks': {
                                        'times': chunk_times[:online_n].tolist(),
                                        'labels': labels_prefix.tolist(),
                                        'label_ids_original': np.unique(labels_prefix).tolist(),
                                        'label_counts': {},
                                        'label_ratios': {},
                                        'label_durations': {},
                                        'total_chunk_duration': float(np.sum(chunk_durs[:online_n])),
                                    },
                                    'segments': None,
                                }

                batch_st += self.batchsize

        embeddings = np.concatenate(embeddings_list, axis=0) if len(embeddings_list) > 0 else np.zeros((0, 0), dtype=np.float32)

        # stage 4: clustering (no oracle speaker_num)
        cluster_labels = cluster_backend(embeddings, speaker_num=None)
        cluster_labels = np.asarray(cluster_labels, dtype=np.int64)

        # Remap labels to [0..K-1] to make downstream stats predictable.
        label_ids_original = np.unique(cluster_labels)
        num_speakers = int(label_ids_original.size)
        if num_speakers == 0:
            warnings_list.append('clustering_returned_empty')
            remapped_labels = cluster_labels
        else:
            remapped_labels = np.searchsorted(label_ids_original, cluster_labels).astype(np.int64)

        chunk_times = np.asarray(chunks, dtype=np.float64)
        chunk_durs = np.maximum(0.0, chunk_times[:, 1] - chunk_times[:, 0])

        if num_speakers > 0:
            chunk_counts = np.bincount(remapped_labels, minlength=num_speakers)
            chunk_dur_by_spk = np.bincount(remapped_labels, weights=chunk_durs, minlength=num_speakers)
            total_chunks = int(chunk_counts.sum())
            total_chunk_dur = float(chunk_dur_by_spk.sum())
            chunk_ratios = (chunk_counts / total_chunks).tolist() if total_chunks > 0 else [0.0] * num_speakers
        else:
            chunk_counts = np.array([], dtype=np.int64)
            chunk_dur_by_spk = np.array([], dtype=np.float64)
            total_chunks = 0
            total_chunk_dur = 0.0
            chunk_ratios = []

        # segment-level distribution (derived from chunk labels)
        segs = None
        speaker_durations = {}
        speaker_turns = {}
        if include_segments and num_speakers > 0:
            segs = [[float(st), float(ed), int(lbl)] for (st, ed), lbl in zip(chunks, remapped_labels.tolist())]
            segs = compressed_seg(segs)
            seg_dur_by_spk = np.zeros(num_speakers, dtype=np.float64)
            seg_turn_by_spk = np.zeros(num_speakers, dtype=np.int64)
            for st, ed, spk in segs:
                seg_dur_by_spk[spk] += max(0.0, float(ed) - float(st))
                seg_turn_by_spk[spk] += 1
            speaker_durations = {str(i): float(seg_dur_by_spk[i]) for i in range(num_speakers)}
            speaker_turns = {str(i): int(seg_turn_by_spk[i]) for i in range(num_speakers)}
        elif include_segments:
            segs = []

        # Embedding diagnostics for heuristic confidence.
        confidence_breakdown: Dict[str, Any] = {
            'num_chunks': int(len(chunks)),
            'num_speakers': num_speakers,
            'count_mode': count_mode,
            'chunk_dur': float(chunk_dur),
            'chunk_step': float(chunk_step),
        }
        confidence = 0.0
        outlier_evidence = None

        if len(chunks) < 8:
            warnings_list.append('too_few_chunks')

        if include_embedding_diagnostics and num_speakers > 0 and embeddings.shape[0] > 0:
            emb = embeddings.astype(np.float32)
            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

            centroids = np.zeros((num_speakers, emb.shape[1]), dtype=np.float32)
            for k in range(num_speakers):
                idx = remapped_labels == k
                if not np.any(idx):
                    continue
                c = emb[idx].mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-12)
                centroids[k] = c

            intra = np.sum(emb * centroids[remapped_labels], axis=1)
            mean_intra = float(np.mean(intra))
            std_intra = float(np.std(intra))

            centroid_sim = centroids @ centroids.T
            if num_speakers > 1:
                mask = ~np.eye(num_speakers, dtype=bool)
                max_inter = float(np.max(centroid_sim[mask]))
                margin = mean_intra - max_inter
                confidence = float(np.clip((margin + 1.0) / 2.0, 0.0, 1.0))
            else:
                max_inter = None
                margin = None
                single_speaker_chunk_support = float(min(1.0, len(chunks) / 24.0))
                confidence = float(np.clip(mean_intra, 0.0, 1.0) * single_speaker_chunk_support)

            dynamic_outlier_threshold = mean_intra - outlier_std_scale * std_intra
            outlier_threshold = dynamic_outlier_threshold if outlier_abs_threshold is None else max(dynamic_outlier_threshold, outlier_abs_threshold)
            outlier_idx = np.where(intra < outlier_threshold)[0]
            outlier_ratio = float(len(outlier_idx) / len(intra))
            suspected_minor = num_speakers == 1 and len(outlier_idx) >= outlier_min_count and outlier_ratio >= outlier_min_ratio

            confidence_breakdown.update({
                'mean_intra_cos': mean_intra,
                'std_intra_cos': std_intra,
                'max_inter_centroid_cos': max_inter,
                'margin': None if margin is None else float(margin),
            })
            if num_speakers == 1:
                confidence_breakdown['single_speaker_chunk_support'] = single_speaker_chunk_support
                if suspected_minor:
                    confidence *= max(0.2, 1.0 - outlier_ratio * 2.0)
                    warnings_list.append('possible_minor_speaker')

            outlier_keep = outlier_idx[:outlier_max_windows]
            outlier_evidence = {
                'detected': suspected_minor,
                'count': int(len(outlier_idx)),
                'ratio': outlier_ratio,
                'threshold': float(outlier_threshold),
                'dynamic_threshold': float(dynamic_outlier_threshold),
                'indices': outlier_keep.tolist(),
                'times': chunk_times[outlier_keep].tolist(),
                'scores': intra[outlier_keep].astype(np.float32).tolist(),
                'params': {
                    'outlier_std_scale': float(outlier_std_scale),
                    'outlier_abs_threshold': outlier_abs_threshold,
                    'outlier_min_count': int(outlier_min_count),
                    'outlier_min_ratio': float(outlier_min_ratio),
                    'outlier_max_windows': int(outlier_max_windows),
                },
            }

            if num_speakers > 1 and margin < 0.1:
                warnings_list.append('low_cluster_separation')

            if return_embeddings:
                confidence_breakdown['centroid_cosine_matrix'] = centroid_sim.astype(np.float32)
        else:
            confidence = 0.0 if num_speakers == 0 else 0.5
            confidence_breakdown['reason'] = 'embedding_diagnostics_disabled_or_unavailable'

        result: Dict[str, Any] = {
            'num_speakers': int(num_speakers),
            'confidence': float(confidence),
            'confidence_breakdown': confidence_breakdown,
            'outlier_evidence': outlier_evidence,
            'warnings': warnings_list,
            'speech': {
                'vad_intervals': vad_time,
                'audio_duration': audio_duration,
                'speech_duration': speech_duration,
                'speech_ratio': speech_ratio,
            },
            'chunks': {
                'times': chunk_times.tolist(),
                'labels': remapped_labels.tolist(),
                'label_ids_original': label_ids_original.tolist(),
                'label_counts': {str(i): int(chunk_counts[i]) for i in range(num_speakers)},
                'label_ratios': {str(i): float(chunk_ratios[i]) for i in range(num_speakers)},
                'label_durations': {str(i): float(chunk_dur_by_spk[i]) for i in range(num_speakers)},
                'total_chunk_duration': float(total_chunk_dur),
            },
        }

        if include_segments:
            result['segments'] = {
                'segs': segs,
                'speaker_durations': speaker_durations,
                'speaker_turns': speaker_turns,
            }

        if return_embeddings:
            result['embeddings'] = embeddings

        return result

    def do_vad(self, wav):
        # wav: [1, T]
        vad_results = self.vad_model(wav[0])[0]
        vad_time = [[vad_t[0]/1000, vad_t[1]/1000] for vad_t in vad_results['value']]
        return vad_time

    def do_segmentation(self, wav):
        # Lazy import: only used when include_overlap=True
        from pyannote.audio import Inference

        segmentations = self.segmentation_model({'waveform': wav, 'sample_rate': self.fs})
        frame_windows = self.segmentation_model.model.receptive_field

        count = Inference.aggregate(
            np.sum(segmentations, axis=-1, keepdims=True),
            frame_windows,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return segmentations, count

    def chunk(self, st, ed, dur=1.5, step=0.75):
        chunks = []
        subseg_st = st
        while subseg_st + dur < ed + step:
            subseg_ed = min(subseg_st + dur, ed)
            chunks.append([subseg_st, subseg_ed])
            subseg_st += step
        return chunks

    def do_emb_extraction(self, chunks, wav):
        # chunks: [[st1, ed1]...]
        # wav: [1, T]
        wavs = [wav[0, int(st*self.fs):int(ed*self.fs)] for st, ed in chunks]
        max_len = max([x.shape[0] for x in wavs])
        wavs = [circle_pad(x, max_len) for x in wavs]
        wavs = torch.stack(wavs).unsqueeze(1)

        embeddings = []
        batch_st = 0
        with torch.no_grad():
            while batch_st < len(chunks):
                wavs_batch = wavs[batch_st: batch_st+self.batchsize].to(self.device)
                feats_batch = torch.vmap(self.feature_extractor)(wavs_batch)
                embeddings_batch = self.embedding_model(feats_batch).cpu()
                embeddings.append(embeddings_batch)
                batch_st += self.batchsize
        embeddings = torch.cat(embeddings, dim=0).numpy()
        return embeddings

    def do_clustering(self, chunks, embeddings, speaker_num=None):
        cluster_labels = self.cluster(
            embeddings, 
            speaker_num = speaker_num if speaker_num is not None else self.speaker_num
        )
        speaker_num = cluster_labels.max()+1
        output_field_labels = [[i[0], i[1], int(j)] for i, j in zip(chunks, cluster_labels)]
        output_field_labels = compressed_seg(output_field_labels)
        return speaker_num, output_field_labels

    def post_process(self, output_field_labels, speaker_num, segmentations, count):
        from scipy import optimize
        num_frames = len(count)
        cluster_frames = np.zeros((num_frames, speaker_num))
        frame_windows = count.sliding_window
        for i in output_field_labels:
            cluster_frames[frame_windows.closest_frame(i[0]+frame_windows.duration/2)\
                :frame_windows.closest_frame(i[1]+frame_windows.duration/2)\
                    , i[2]] = 1.0

        activations = np.zeros((num_frames, speaker_num))
        num_chunks, num_frames_per_chunk, num_classes = segmentations.data.shape
        for i, (c, data) in enumerate(segmentations):
            # data: [num_frames_per_chunk, num_classes]
            # chunk_cluster_frames: [num_frames_per_chunk, speaker_num]
            start_frame = frame_windows.closest_frame(c.start+frame_windows.duration/2)
            end_frame = start_frame + num_frames_per_chunk
            chunk_cluster_frames = cluster_frames[start_frame:end_frame]
            align_chunk_cluster_frames = np.zeros((num_frames_per_chunk, speaker_num))

            # assign label to each dimension of "data" according to number of 
            # overlap frames between "data" and "chunk_cluster_frames"
            cost_matrix = []
            for j in range(num_classes):
                if sum(data[:, j])>0:
                    num_of_overlap_frames = [(data[:, j].astype('int') & d.astype('int')).sum() \
                        for d in chunk_cluster_frames.T]
                else:
                    num_of_overlap_frames = [-1]*speaker_num
                cost_matrix.append(num_of_overlap_frames)
            cost_matrix = np.array(cost_matrix) # (num_classes, speaker_num)
            row_index, col_index = optimize.linear_sum_assignment(-cost_matrix)
            for j in range(len(row_index)):
                r = row_index[j]
                c = col_index[j]
                if cost_matrix[r, c] > 0:
                    align_chunk_cluster_frames[:, c] = np.maximum(
                            data[:, r], align_chunk_cluster_frames[:, c]
                            )
            activations[start_frame:end_frame] += align_chunk_cluster_frames

        # correct activations according to count_data
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations)
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            cur_max_spk_num = min(speaker_num, c.item())
            for i in range(cur_max_spk_num):
                if activations[t, speakers[i]] > 0:
                    binary[t, speakers[i]] = 1.0

        supplement_field = (binary.sum(-1)==0) & (cluster_frames.sum(-1)!=0)
        binary[supplement_field] = cluster_frames[supplement_field]
        return binary

    def binary_to_segs(self, binary, timestamps, threshold=0.5):
        output_field_labels = []
        # binary: [num_frames, num_classes]
        # timestamps: [T_1, ..., T_num_frames]
        if len(timestamps) == 0:
            return []
        last_t = timestamps[-1]

        for k, k_scores in enumerate(binary.T):
            start = timestamps[0]
            is_active = k_scores[0] > threshold

            for t, y in zip(timestamps[1:], k_scores[1:]):
                if is_active:
                    if y < threshold:
                        output_field_labels.append([round(start, 3), round(t, 3), k])
                        start = t
                        is_active = False
                else:
                    if y > threshold:
                        start = t
                        is_active = True

            if is_active:
                output_field_labels.append([round(start, 3), round(last_t, 3), k])
        return sorted(output_field_labels, key=lambda x: x[0])

    def save_diar_output(self, out_file, wav_id=None, output_field_labels=None):
        if output_field_labels is None and self.output_field_labels is None:
            raise ValueError('No results can be saved.')
        if output_field_labels is None:
            output_field_labels = self.output_field_labels

        wav_id = 'default' if wav_id is None else wav_id
        if out_file.endswith('rttm'):
            line_str ="SPEAKER {} 0 {:.3f} {:.3f} <NA> <NA> {:d} <NA> <NA>\n"
            with open(out_file, 'w') as f:
                for seg in output_field_labels:
                    seg_st, seg_ed, cluster_id = seg
                    f.write(line_str.format(wav_id, seg_st, seg_ed-seg_st, cluster_id))
        elif out_file.endswith('json'):
            out_json = {}
            for seg in output_field_labels:
                seg_st, seg_ed, cluster_id = seg
                item = {
                    'start': seg_st,
                    'stop': seg_ed,
                    'speaker': cluster_id,
                }
                segid = wav_id+'_'+str(round(seg_st, 3))+\
                    '_'+str(round(seg_ed, 3))
                out_json[segid] = item
            with open(out_file, mode='w') as f:
                json.dump(out_json, f, indent=2)
        else:
            raise ValueError('The supported output file formats are currently limited to RTTM and JSON.')

    def normalize_device(self, device=None):
        if device is None:
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        else:
            assert isinstance(device, torch.device)
        return device

def get_valid_field(count):
    valid_field = []
    start = None
    for i, (c, data) in enumerate(count):
        if data.item()==0 or i==len(count)-1:
            if start is not None:
                end = c.middle
                valid_field.append([start, end])
                start = None
        else:
            if start is None:
                start = c.middle
    return valid_field

def compressed_seg(seg_list):
    new_seg_list = []
    for i, seg in enumerate(seg_list):
        seg_st, seg_ed, cluster_id = seg
        if i == 0:
            new_seg_list.append([seg_st, seg_ed, cluster_id])
        elif cluster_id == new_seg_list[-1][2]:
            if seg_st > new_seg_list[-1][1]:
                new_seg_list.append([seg_st, seg_ed, cluster_id])
            else:
                new_seg_list[-1][1] = seg_ed
        else:
            if seg_st < new_seg_list[-1][1]:
                p = (new_seg_list[-1][1]+seg_st) / 2
                new_seg_list[-1][1] = p
                seg_st = p
            new_seg_list.append([seg_st, seg_ed, cluster_id])
    return new_seg_list

def main_process(rank, nprocs, args, wav_list):
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        ngpus = torch.cuda.device_count()
        device = torch.device('cuda:%d'%(rank%ngpus))
    diarization = Diarization3Dspeaker(device, args.include_overlap, args.hf_access_token, args.speaker_num)
    
    wav_list = wav_list[rank::nprocs]
    if rank == 0 and (not args.diable_progress_bar):
        wav_list = tqdm(wav_list, desc=f"Rank 0 processing")
    for wav_path in wav_list:
        output = diarization(wav_path)
        # write to file
        wav_id = os.path.basename(wav_path).rsplit('.', 1)[0]
        if args.out_dir is not None:
            out_file = os.path.join(args.out_dir, wav_id + '.%s' % args.out_type)
        else:
            out_file = '%s.%s' % (wav_path.rsplit('.', 1)[0], args.out_type)
        diarization.save_diar_output(out_file, wav_id, output_field_labels=output)

def main():
    args = parser.parse_args()
    if args.include_overlap and args.hf_access_token is None:
        parser.error("--hf_access_token is required when --include_overlap is specified.")
    
    get_speaker_embedding_model()
    get_voice_activity_detection_model()
    get_cluster_backend()
    if args.include_overlap:
        get_segmentation_model(args.hf_access_token)
    print(f'[INFO]: Model downloaded successfully.')

    if args.wav.endswith('.wav'):
        # input is a wav file
        wav_list = [args.wav]
    else:
        try:
            # input should be a wav list
            with open(args.wav,'r') as f:
                wav_list = [i.strip() for i in f.readlines()]
        except:
            raise Exception('[ERROR]: Input should be a wav file or a wav list.')
    assert len(wav_list) > 0

    if args.nprocs is None:
        ngpus = torch.cuda.device_count()
        if ngpus > 0:
            print(f'[INFO]: Detected {ngpus} GPUs.')
            args.nprocs = ngpus
        else:
            print('[INFO]: No GPUs detected.')
            args.nprocs = 1

    args.nprocs = min(len(wav_list), args.nprocs)
    print(f'[INFO]: Set {args.nprocs} processes to extract embeddings.')

    # output dir
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)

    mp.spawn(main_process, nprocs=args.nprocs, args=(args.nprocs, args, wav_list))

if __name__ == '__main__':
    main()
