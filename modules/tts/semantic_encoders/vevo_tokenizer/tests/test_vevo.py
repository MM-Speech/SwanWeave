from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from user.vevo.audio import prepare_waveform
from user.vevo.checkpoint import (
    resolve_content_checkpoint_path,
    resolve_content_style_checkpoint_path,
)
from user.vevo.extractor import (
    VevoTokenExtractor,
    VevoTokenResult,
    _AudioRecord,
    _ChunkRecord,
    _duration_reduce,
    run_vevo_token_model,
)


class _FakeContentCodebook:
    def __init__(self, token_ids: torch.Tensor):
        self._token_ids = token_ids

    def forward_index(self, _z: torch.Tensor):
        return None, self._token_ids


class _FakeContentQuantizer:
    def __init__(self, token_ids: torch.Tensor):
        self.codebook = _FakeContentCodebook(token_ids)


class _FakeContentTokenizer:
    def __init__(self, token_ids: torch.Tensor):
        self.quantizer = _FakeContentQuantizer(token_ids)

    def encoder(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def projector(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _FakeContentStyleTokenizer:
    def __init__(self, token_ids: torch.Tensor):
        self._token_ids = token_ids

    def quantize(self, _x: torch.Tensor):
        embedding = torch.zeros(self._token_ids.shape[0], self._token_ids.shape[1], 4)
        return self._token_ids, embedding


class VevoExtractorTests(unittest.TestCase):
    def test_duration_reduce_matches_expected_sequence(self):
        tokens = torch.tensor([1, 1, 2, 2, 3, 3, 4], dtype=torch.long)
        reduced = _duration_reduce(tokens, n_gram=1)
        self.assertTrue(torch.equal(reduced, torch.tensor([1, 2, 3, 4], dtype=torch.long)))

    def test_prepare_waveform_handles_multichannel_input(self):
        waveform = np.stack(
            [
                np.linspace(-1.0, 1.0, 4800, dtype=np.float32),
                np.linspace(1.0, -1.0, 4800, dtype=np.float32),
            ],
            axis=0,
        )
        prepared = prepare_waveform(
            waveform,
            sample_rate=48000,
            device=torch.device("cpu"),
        )
        self.assertEqual(prepared.waveform_24k.ndim, 2)
        self.assertEqual(prepared.waveform_16k.ndim, 2)
        self.assertEqual(prepared.source_sample_rate, 48000)

    def test_checkpoint_resolution_prefers_explicit_runtime_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            content_dir = temp_path / "content"
            content_dir.mkdir()
            (content_dir / "hubert_large_l18_c32.pkl").touch()

            style_dir = temp_path / "style"
            style_dir.mkdir()
            (style_dir / "model.safetensors").touch()

            self.assertEqual(
                resolve_content_checkpoint_path(content_dir),
                content_dir / "hubert_large_l18_c32.pkl",
            )
            self.assertEqual(
                resolve_content_style_checkpoint_path(style_dir),
                style_dir / "model.safetensors",
            )

    def test_private_hubert_token_extraction_returns_1d_int64(self):
        extractor = object.__new__(VevoTokenExtractor)
        extractor.content_tokenizer = _FakeContentTokenizer(
            torch.tensor([[[1, 1, 2, 2, 3]]], dtype=torch.long)
        )
        extractor.content_style_tokenizer = _FakeContentStyleTokenizer(
            torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
        )
        extractor.hubert_feat_norm_mean = torch.zeros(3)
        extractor.hubert_feat_norm_std = torch.ones(3)

        feats = torch.randn(1, 5, 3)
        feat_lengths = torch.tensor([5], dtype=torch.long)

        content_ids = extractor._extract_content_ids_from_hubert(feats, feat_lengths)
        content_style_ids = extractor._extract_content_style_ids_from_hubert(feats, feat_lengths)

        self.assertEqual(content_ids.dtype, np.int64)
        self.assertEqual(content_style_ids.dtype, np.int64)
        self.assertEqual(content_ids.ndim, 1)
        self.assertEqual(content_style_ids.ndim, 1)
        np.testing.assert_array_equal(content_ids, np.array([1, 2, 3], dtype=np.int64))
        np.testing.assert_array_equal(
            content_style_ids,
            np.array([7, 8, 9, 10], dtype=np.int64),
        )

    def test_chunk_builder_uses_lookahead_and_tail_padding(self):
        extractor = object.__new__(VevoTokenExtractor)
        waveform = torch.arange(144000, dtype=torch.float32)
        audio_record = _AudioRecord(
            audio_index=0,
            source_sample_rate=16000,
            waveform_16k=waveform,
        )

        chunk_records = extractor._build_chunk_records_for_audio(audio_record)
        self.assertEqual(len(chunk_records), 2)

        self.assertEqual(chunk_records[0].core_len_16k, 128000)
        self.assertEqual(chunk_records[0].infer_len_16k, 128080)
        self.assertEqual(chunk_records[0].target_token_len, 400)
        self.assertTrue(
            torch.equal(
                chunk_records[0].waveform_16k[-80:],
                waveform[128000:128080],
            )
        )

        self.assertEqual(chunk_records[1].core_len_16k, 16000)
        self.assertEqual(chunk_records[1].infer_len_16k, 16080)
        self.assertEqual(chunk_records[1].target_token_len, 50)
        self.assertTrue(torch.all(chunk_records[1].waveform_16k[-80:] == 0))

    def test_chunk_builder_pads_last_full_chunk_to_400_tokens(self):
        extractor = object.__new__(VevoTokenExtractor)
        waveform = torch.ones(128000, dtype=torch.float32)
        audio_record = _AudioRecord(
            audio_index=0,
            source_sample_rate=16000,
            waveform_16k=waveform,
        )

        chunk_records = extractor._build_chunk_records_for_audio(audio_record)
        self.assertEqual(len(chunk_records), 1)
        self.assertEqual(chunk_records[0].infer_len_16k, 128080)
        self.assertEqual(chunk_records[0].target_token_len, 400)
        self.assertTrue(torch.all(chunk_records[0].waveform_16k[-80:] == 0))

    def test_normalize_audio_inputs_rejects_mixed_lists(self):
        extractor = object.__new__(VevoTokenExtractor)
        with self.assertRaisesRegex(ValueError, "homogenous"):
            extractor._normalize_audio_inputs(["a.wav", np.zeros(10, dtype=np.float32)])

    def test_normalize_audio_inputs_requires_sample_rates_for_waveform_lists(self):
        extractor = object.__new__(VevoTokenExtractor)
        with self.assertRaisesRegex(ValueError, "sample_rates is required"):
            extractor._normalize_audio_inputs([np.zeros(10, dtype=np.float32)])

    def test_extract_chunk_tokens_only_runs_requested_branch(self):
        extractor = object.__new__(VevoTokenExtractor)
        extractor.device = torch.device("cpu")
        call_count = {"content": 0, "content_style": 0}

        def _fake_hubert_features(_wavs, wav_lens):
            feats = torch.zeros(wav_lens.shape[0], 2, 3)
            feat_lengths = torch.tensor([2] * wav_lens.shape[0], dtype=torch.long)
            return feats, feat_lengths

        def _fake_content(_feats, token_lengths):
            call_count["content"] += 1
            return [np.arange(int(length), dtype=np.int64) for length in token_lengths.tolist()]

        def _fake_content_style(_feats, token_lengths):
            call_count["content_style"] += 1
            return [np.arange(int(length), dtype=np.int64) for length in token_lengths.tolist()]

        extractor._extract_hubert_features = _fake_hubert_features
        extractor._extract_content_ids_from_hubert_batch = _fake_content
        extractor._extract_content_style_ids_from_hubert_batch = _fake_content_style

        chunk_record = _ChunkRecord(
            audio_index=0,
            chunk_index=0,
            core_len_16k=640,
            infer_len_16k=720,
            target_token_len=2,
            waveform_16k=torch.zeros(720, dtype=torch.float32),
        )
        content_chunks, content_style_chunks = extractor._extract_chunk_tokens(
            [chunk_record],
            num_audios=1,
            vector_type="content",
            batch_size=4,
        )

        self.assertEqual(call_count["content"], 1)
        self.assertEqual(call_count["content_style"], 0)
        self.assertIsNotNone(content_chunks)
        self.assertIsNone(content_style_chunks)
        np.testing.assert_array_equal(content_chunks[0][0], np.array([0, 1], dtype=np.int64))

    def test_extract_from_audio_records_reduces_content_after_chunk_concat(self):
        extractor = object.__new__(VevoTokenExtractor)

        def _fake_build_chunk_records(_audio_record):
            return [
                _ChunkRecord(
                    audio_index=0,
                    chunk_index=0,
                    core_len_16k=128000,
                    infer_len_16k=128080,
                    target_token_len=400,
                    waveform_16k=torch.zeros(128080),
                ),
                _ChunkRecord(
                    audio_index=0,
                    chunk_index=1,
                    core_len_16k=128000,
                    infer_len_16k=128080,
                    target_token_len=400,
                    waveform_16k=torch.zeros(128080),
                ),
            ]

        def _fake_extract_chunk_tokens(
            chunk_records,
            num_audios,
            vector_type,
            batch_size,
        ):
            self.assertEqual(len(chunk_records), 2)
            self.assertEqual(num_audios, 1)
            self.assertEqual(vector_type, "content")
            self.assertEqual(batch_size, 8)
            return [[np.array([7, 7], dtype=np.int64), np.array([7, 7], dtype=np.int64)]], None

        extractor._build_chunk_records_for_audio = _fake_build_chunk_records
        extractor._extract_chunk_tokens = _fake_extract_chunk_tokens

        audio_record = _AudioRecord(
            audio_index=0,
            source_sample_rate=16000,
            waveform_16k=torch.ones(256000),
        )
        result = extractor._extract_from_audio_records(
            [audio_record],
            vector_type="content",
            batch_size=8,
            reduce_content=True,
        )[0]

        np.testing.assert_array_equal(result.content_ids, np.array([7], dtype=np.int64))
        self.assertIsNone(result.content_style_ids)

    def test_run_vevo_token_model_routes_batch_tensor_to_extract_batch(self):
        class _FakeModel:
            def __init__(self):
                self.batch_called = False

            def extract_batch(self, audio, **kwargs):
                self.batch_called = True
                self.audio = audio
                self.kwargs = kwargs
                return "batch"

            def extract_from_waveform(self, *_args, **_kwargs):
                raise AssertionError("single-waveform path should not be used")

        model = _FakeModel()
        batch_audio = np.zeros((2, 160), dtype=np.float32)
        output = run_vevo_token_model(
            model,
            batch_audio,
            sample_rate=16000,
            audio_lengths=[160, 120],
            batch_size=4,
        )
        self.assertEqual(output, "batch")
        self.assertTrue(model.batch_called)
        self.assertEqual(model.kwargs["batch_size"], 4)

    def test_run_vevo_token_model_keeps_single_2d_waveform_semantics_without_lengths(self):
        class _FakeModel:
            def __init__(self):
                self.single_called = False

            def extract_from_waveform(self, audio, **kwargs):
                self.single_called = True
                self.audio = audio
                self.kwargs = kwargs
                return "single"

            def extract_batch(self, *_args, **_kwargs):
                raise AssertionError("batch path should not be used")

        model = _FakeModel()
        waveform = np.zeros((2, 160), dtype=np.float32)
        output = run_vevo_token_model(model, waveform, sample_rate=16000)
        self.assertEqual(output, "single")
        self.assertTrue(model.single_called)
        self.assertEqual(model.kwargs["sample_rate"], 16000)

    def test_result_to_npz_roundtrip(self):
        result = VevoTokenResult(
            content_ids=np.array([1, 2, 3], dtype=np.int64),
            content_style_ids=np.array([4, 5, 6], dtype=np.int64),
            source_sample_rate=44100,
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "tokens.npz"
            result.to_npz(output_path)
            saved = np.load(output_path)
            np.testing.assert_array_equal(saved["content_ids"], result.content_ids)
            np.testing.assert_array_equal(saved["content_style_ids"], result.content_style_ids)
            self.assertEqual(int(saved["source_sample_rate"]), 44100)
            self.assertEqual(str(saved["vector_type"]), "both")


if __name__ == "__main__":
    unittest.main()
