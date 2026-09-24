"""Tests for the subject-free speech + music route (synthetic data only; no dataset files needed)."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app')); sys.path.insert(0, str(ROOT / 'app/src'))
_cache_name = os.environ.get('ALIGNED_TARGET_CACHE_NAME')
import universal_train as ut
from aligned_recovery_eval import evaluate_recovery
from aligned_recovery_model import recovery_loss
from eeg2speech.aligned import EEG_SAMPLES, EEG_START
import music as md
import music as mio
import universal_model as um
sa = um
if _cache_name is None:
    os.environ.pop('ALIGNED_TARGET_CACHE_NAME', None)
else:
    os.environ['ALIGNED_TARGET_CACHE_NAME'] = _cache_name


def montage_xyz(name='biosemi64'):
    import mne
    return np.array(list(mne.channels.make_standard_montage(name).get_positions()['ch_pos'].values()))


def tiny_decoder(dimension=12, mel_bins=80, teacher_frames=199, mel_frames=251, window=4.):
    return um.ConfigurableAcousticDecoder(torch.linspace(.0125, window - .0275, teacher_frames), torch.linspace(0, window, mel_frames),
                                          speech_dimension=dimension, hidden=16, layers=2, mel_bins=mel_bins)


def decoder_payload(decoder):
    spec = dict(speech_times=decoder.speech_times.tolist(), mel_times=decoder.mel_times.tolist(),
                speech_dimension=decoder.normalizer.mean.numel(), hidden=16, layers=2, mel_bins=decoder.mel_bins)
    return dict(decoder_spec=spec, decoder={k: v.clone() for k, v in decoder.state_dict().items()})


def tiny_model(**kw):
    torch.manual_seed(0)
    return um.UniversalEEGModel({'speech': tiny_decoder(), 'music': tiny_decoder(8, 100, 299, 375)},
                                virtual=8, width=16, harmonics=4, **kw)


# --- synthetic DS004940-like speech data ---------------------------------------------------

class FakeSpeech:
    """Reads through ``frame.rid`` so aligned_recovery.subset() copies stay consistent (as AlignedDataset does)."""
    def __init__(self, subjects, contents, channels=16, seed=0):
        rng = np.random.default_rng(seed)
        xyz = montage_xyz('biosemi128')[:channels].astype(np.float32)
        rows, self.records = [], []
        for s in subjects:
            for c in contents:
                rid = len(self.records)
                mask = np.ones(channels, bool); mask[rid % channels] = rid % 3 != 0
                frames = 140 + 7 * (ord(c[-1]) % 9)
                t = np.arange(64000) / 16000
                wave = rng.normal(size=64000) * (1 + np.sin(2 * np.pi * (2 + ord(c[-1]) % 3) * t)) * (t < frames * 256 / 16000)
                self.records.append(dict(
                    eeg=torch.from_numpy((rng.normal(size=(channels, EEG_SAMPLES)) * mask[:, None]).astype(np.float32)),
                    channel_xyz=torch.from_numpy(xyz), channel_mask=torch.from_numpy(mask),
                    time_mask=torch.ones(EEG_SAMPLES, dtype=torch.bool), teacher=torch.randn(199, 12, generator=torch.Generator().manual_seed(ord(c[-1]))),
                    mel=torch.randn(80, 251, generator=torch.Generator().manual_seed(100 + ord(c[-1]))),
                    oracle_duration_frames=torch.tensor(frames), wave=torch.from_numpy(wave.astype(np.float32)),
                    content=c, subject=s, trial_id=f'{s}-{c}'))
                rows.append(dict(trial_id=f'{s}-{c}', subject=s, content_group=c, stimulus_duration_seconds=frames * 256 / 16000,
                                 task='N400Active', rid=rid))
        self.frame = pd.DataFrame(rows)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        return self.records[int(self.frame.rid.iat[i])]


class FakeBank:
    def __init__(self, dataset):
        self.eeg = torch.stack([dataset[i]['eeg'] for i in range(len(dataset))]).half()
        self.mask = torch.stack([dataset[i]['channel_mask'] for i in range(len(dataset))])

    def __getitem__(self, i):
        return self.eeg[i].float(), self.mask[i]


def templates():
    return tuple(torch.zeros(80, 251) for _ in range(4))


# --- synthetic prepared music dataset --------------------------------------------------------

def make_music(folder, pieces=(1, 2, 3), subjects=('sub-01', 'sub-02', 'sub-03'), duration=12., channels=8):
    """Shards whose EEG value = seconds since piece onset + 100 * channel, and targets whose first dimension = time."""
    folder = Path(folder); (folder / 'shards').mkdir(parents=True)
    xyz = montage_xyz('biosemi64')[:channels].astype(np.float32)
    names = [f'E{i}' for i in range(channels)]
    onset, rate = 256, 256
    samples = onset + int((duration + 1.) * rate)
    rows = []
    for s_index, subject in enumerate(subjects):
        path = folder / 'shards' / f'{subject}.h5'
        with h5py.File(path, 'w') as h5:
            h5.attrs.update(contract=md.CONTRACT, channel_order_hash='hash', sfreq=rate)
            h5.create_dataset('channel_xyz', data=xyz)
            for k, piece in enumerate(pieces):
                seconds = (np.arange(samples) - onset) / rate
                data = seconds[None, :] + 100. * np.arange(channels)[:, None] + 1000. * piece
                h5.create_dataset(f'trials/{k:02d}', data=data.astype(np.float32))
                h5.create_dataset(f'trials_valid/{k:02d}', data=np.ones(channels, bool))
                rows.append(dict(dataset='toy', subject=subject, group='g', trial=k, piece=piece, repeat=1,
                                 shard_path=str(path), onset_sample=onset, n_samples=samples, stimulus_duration_s=duration,
                                 content_role='validation' if piece == pieces[-1] else 'train',
                                 subject_role='heldout' if s_index == len(subjects) - 1 else 'train'))
    manifest = folder / 'manifest.csv'; pd.DataFrame(rows).to_csv(manifest, index=False)
    normalizer = folder / 'normalizer.json'
    normalizer.write_text(json.dumps(dict(center=[0.] * channels, scale=[1.] * channels, channel_order_hash='hash')))
    targets = folder / 'targets.h5'
    with h5py.File(targets, 'w') as h5:
        h5.attrs.update(contract=md.TARGET_CONTRACT, teacher='toy', teacher_layer=-1)
        h5.create_dataset('grid/teacher_times', data=((np.arange(299) + .5) * 320 / 24000).astype(np.float32))
        h5.create_dataset('grid/mel_times', data=mio.bigvgan_mel_times(375).astype(np.float32))
        for piece in pieces:
            g = h5.create_group(f'pieces/{piece}')
            teacher_times = (np.arange(int(duration * 75)) + .5) / 75
            teacher = np.random.default_rng(piece).normal(size=(len(teacher_times), 8)); teacher[:, 0] = teacher_times
            mel_times = mio.bigvgan_mel_times(int(duration * 93.75))
            mel = np.random.default_rng(10 + piece).normal(size=(100, len(mel_times))); mel[0] = mel_times
            acoustic = np.random.default_rng(20 + piece).normal(size=(2, int(duration * 64))); acoustic[0] = np.arange(acoustic.shape[1]) / 64
            g.create_dataset('teacher', data=teacher.astype(np.float32)); g.create_dataset('teacher_times', data=teacher_times)
            g.create_dataset('mel', data=mel.astype(np.float32)); g.create_dataset('mel_times', data=mel_times)
            g.create_dataset('acoustic', data=acoustic.astype(np.float32)); g.attrs['duration_s'] = duration
    return manifest, targets, normalizer


class SplineAndAugmentationTests(unittest.TestCase):
    def test_spline_matches_mne_and_interpolates_smooth_fields(self):
        from mne.channels.interpolation import _make_interpolation_matrix
        xyz = montage_xyz()
        np.testing.assert_allclose(sa.spherical_spline(xyz[:48], xyz[48:], alpha=1e-5), _make_interpolation_matrix(xyz[:48], xyz[48:]),
                                   atol=1e-8)
        alpha = sa.SpatialConfig().spline_alpha
        field = lambda p: (p / np.linalg.norm(p, axis=1, keepdims=True)) @ np.array([.3, -.5, .8])
        moved = xyz @ sa.rotation(np.radians(5), np.radians(3), 0).T
        estimate = sa.spherical_spline(xyz, moved, alpha=alpha) @ field(xyz)
        self.assertLess(np.abs(estimate - field(moved)).max() / np.abs(field(moved)).max(), .01)
        # The augmentation's regularisation must not blur: the self-map is (nearly) the identity...
        self.assertGreater(float(np.diag(sa.spherical_spline(xyz, xyz, alpha=alpha)).min()), .95)
        # ...whereas MNE's default keeps well under half of each channel.
        self.assertLess(float(np.diag(sa.spherical_spline(xyz, xyz, alpha=1e-5)).mean()), .5)

    def test_augmenter_identity_masks_rms_and_determinism(self):
        xyz = torch.from_numpy(montage_xyz().astype(np.float32))
        eeg = torch.randn(6, 64, 300); mask = torch.ones(6, 64, dtype=torch.bool); mask[:, 5] = False; mask[2, 10:20] = False
        eeg = eeg * mask[:, :, None]
        off = sa.SpatialAugmenter(sa.SpatialConfig())
        out, out_mask = off(eeg, mask, xyz)
        self.assertIs(out, eeg); self.assertIs(out_mask, mask)
        cfg = sa.SpatialConfig(geometry=1, reference=1, conduction=1, subset=1, keep_low=.5, keep_high=.5, regional=0)
        out, out_mask = sa.SpatialAugmenter(cfg)(eeg, mask, xyz, rng=np.random.default_rng(3))
        again, again_mask = sa.SpatialAugmenter(cfg)(eeg, mask, xyz, rng=np.random.default_rng(3))
        torch.testing.assert_close(out, again); self.assertTrue(torch.equal(out_mask, again_mask))
        self.assertEqual(out.shape, eeg.shape)
        self.assertFalse(bool((out_mask & ~mask).any()))                          # never revives an invalid channel
        self.assertEqual(float(out[~out_mask].abs().sum()), 0.)
        for b in range(6):
            self.assertLessEqual(abs(int(out_mask[b].sum()) - int(mask[b].sum()) // 2), 2)
            before = eeg[b][mask[b]].square().mean(); after = out[b][out_mask[b]].square().mean()
            torch.testing.assert_close(after, before, rtol=1e-4, atol=1e-6)
        refill = sa.SpatialAugmenter(sa.SpatialConfig(subset=1, keep_low=.5, keep_high=.5, regional=0, subset_mode='interpolate'))
        out, out_mask = refill(eeg, mask, xyz, rng=np.random.default_rng(1))
        self.assertTrue(torch.equal(out_mask, mask)); self.assertGreater(float(out[mask].abs().min()), 0.)
        np.testing.assert_allclose(sa.laplacian_matrix(montage_xyz()).sum(1), 0, atol=1e-12)

    def test_recolouring_and_covariances(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=(12, 12)); b = rng.normal(size=(12, 12))
        ca, cb = a @ a.T + np.eye(12), b @ b.T + np.eye(12)
        w = sa.recolour_matrix(ca, cb, 1.)
        np.testing.assert_allclose(w @ ca @ w.T, cb, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(sa.recolour_matrix(ca, cb, 0.), np.eye(12))
        trials = np.einsum('ij,njt->nit', np.linalg.cholesky(ca), rng.normal(size=(200, 12, 400)))
        masks = np.ones((200, 12), bool); masks[::3, 4] = False
        estimate = sa.estimate_covariance(trials * masks[:, :, None], masks, shrinkage=0.)
        self.assertLess(np.abs(estimate - ca).max() / np.abs(ca).max(), .05)
        xyz = torch.from_numpy(montage_xyz()[:12].astype(np.float32))
        aug = sa.SpatialAugmenter(sa.SpatialConfig(recolour=1, recolour_low=1, recolour_high=1), {'d:a': ca, 'd:b': cb, 'e:c': ca})
        eeg = torch.randn(2, 12, 50); mask = torch.ones(2, 12, dtype=torch.bool)
        out, _ = aug(eeg, mask, xyz, groups=['d:a', 'e:c'], rng=np.random.default_rng(0))
        self.assertFalse(torch.allclose(out[0], eeg[0]))                          # recoloured towards d:b
        torch.testing.assert_close(out[1], eeg[1])                                  # no other group in dataset e
        first, second = sa.split_views(torch.tensor([[True] * 7 + [False]]), np.random.default_rng(0))
        self.assertFalse(bool((first & second).any())); self.assertEqual(int((first | second).sum()), 7)


class ModelTests(unittest.TestCase):
    def test_subject_free_montage_agnostic_and_invariant(self):
        model = tiny_model().eval()
        self.assertFalse(any('subject' in name for name, _ in model.named_parameters()))
        for channels in (64, 128):
            xyz = torch.from_numpy(montage_xyz(f'biosemi{channels}').astype(np.float32))
            eeg = torch.randn(3, channels, EEG_SAMPLES); mask = torch.ones(3, channels, dtype=torch.bool); mask[0, 3] = False
            time = torch.ones(3, EEG_SAMPLES, dtype=torch.bool)
            state = model(eeg, xyz, mask, time)
            self.assertEqual(tuple(state.aligned_sequence.shape), (3, 199, 12)); self.assertEqual(tuple(state.native_mel.shape), (3, 80, 251))
            self.assertEqual(tuple(state.acoustic.shape), (3, 2, um.ACOUSTIC_FRAMES))
            music = model(eeg, xyz, mask, time, domain='music')
            self.assertEqual(tuple(music.native_mel.shape), (3, 100, 375))
            order = torch.randperm(channels)
            permuted = model(eeg[:, order], xyz[order], mask[:, order], time)
            torch.testing.assert_close(permuted.native_mel, state.native_mel, atol=1e-5, rtol=1e-4)
            changed = eeg.clone(); changed[0, 3] = 99.
            torch.testing.assert_close(model(changed, xyz, mask, time).native_mel, state.native_mel)
            torch.testing.assert_close(model(3 * eeg, xyz, mask, time).native_mel, state.native_mel, atol=1e-5, rtol=1e-4)
        with self.assertRaises(ValueError):
            model(eeg, xyz, mask, time, torch.zeros(3, dtype=torch.long))

    def test_positional_code_only_for_locked_trials_and_frozen_decoders(self):
        model = tiny_model()
        with torch.no_grad():
            model.position.normal_()
        xyz = torch.from_numpy(montage_xyz().astype(np.float32))
        eeg = torch.randn(2, 64, EEG_SAMPLES); mask = torch.ones(2, 64, dtype=torch.bool); time = torch.ones(2, EEG_SAMPLES, dtype=torch.bool)
        model.eval()
        locked = model(eeg, xyz, mask, time, locked=torch.tensor([True, False]))
        free = model(eeg, xyz, mask, time, locked=torch.tensor([False, False]))
        self.assertFalse(torch.allclose(locked.native_mel[0], free.native_mel[0])); torch.testing.assert_close(locked.native_mel[1], free.native_mel[1])
        model.train(); self.assertFalse(model.decoders.training)
        model(eeg, xyz, mask, time).native_mel.mean().backward()
        self.assertTrue(all(p.grad is None for p in model.decoders.parameters()))
        self.assertIsNotNone(model.spatial_attention.logits.weight.grad)

    def test_trunk_loading_skips_physical_spatial_filters(self):
        model = tiny_model(); source = tiny_model()
        with torch.no_grad():
            for p in source.parameters():
                p.add_(1.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'trunk.pt'
            torch.save(dict(trunk={k: v for k, v in source.state_dict().items() if k.split('.')[0] in ('spatial', 'temporal', 'blocks', 'output_norm')}), path)
            before = model.spatial.weight.clone()
            loaded = um.load_trunk(model, path)
        self.assertTrue(all(k.split('.')[0] in um.TRUNK_KEYS for k in loaded))
        torch.testing.assert_close(model.temporal.weight, source.temporal.weight); torch.testing.assert_close(model.spatial.weight, before)

    def test_v3_evaluation_runs_unchanged_on_the_universal_model(self):
        data = FakeSpeech(['s1', 's2'], ['a', 'b', 'c'])
        report = evaluate_recovery(tiny_model(), data, data, torch.device('cpu'), 4, None, bootstrap=False, templates=templates())
        for key in ('native_mel_mae', 'wrong_trial_gain', 'zero_gain', 'time_block_shuffle_gain', 'retrieval_mrr', 'envelope_corr'):
            self.assertTrue(np.isfinite(report[key]), key)


class LossTests(unittest.TestCase):
    def test_acoustic_targets_track_the_modulator_at_both_rates(self):
        for rate in (16000, 24000):
            t = np.arange(4 * rate) / rate
            modulator = 1.2 + np.sin(2 * np.pi * 3 * t)
            wave = torch.from_numpy((np.random.default_rng(0).normal(size=len(t)) * modulator).astype(np.float32))
            out = um.acoustic_targets(wave[None], rate)
            self.assertEqual(tuple(out.shape), (1, 2, um.ACOUSTIC_FRAMES))
            reference = np.log(modulator[::rate // 64][:um.ACOUSTIC_FRAMES] ** 2)
            self.assertGreater(np.corrcoef(out[0, 0].numpy(), reference)[0, 1], .9)
        clicks = np.zeros(4 * 16000, np.float32)
        for s in (1., 2.5):
            clicks[int(s * 16000):int(s * 16000) + 80] = 1.
        onset = um.acoustic_targets(torch.from_numpy(clicks)[None], 16000)[0, 1].numpy()
        peaks = sorted(np.argsort(onset)[-2:])
        self.assertLessEqual(abs(peaks[0] - 64), 1); self.assertLessEqual(abs(peaks[1] - 160), 1)

    def test_correlation_loss_and_masks(self):
        x = torch.randn(3, 2, 50)
        loss, r = um.correlation_loss(x, x)
        self.assertLess(float(loss), 1e-5); torch.testing.assert_close(r, torch.ones(3, 2), atol=1e-5, rtol=0)
        mask = torch.ones(3, 50, dtype=torch.bool); mask[:, 40:] = False
        y = x.clone(); y[..., 40:] = 50.
        torch.testing.assert_close(um.correlation_loss(x, y, mask)[1], um.correlation_loss(x, x, mask)[1])

    def test_aligned_loss_matches_v3_and_respects_negatives(self):
        decoder = tiny_decoder()
        state = um.UniversalState(torch.randn(3, 199, 12), decoder.speech_times, torch.randn(3, 80, 251), decoder.mel_times,
                                  torch.randn(3, 12), torch.rand(3), None)
        teacher, mel = torch.randn(3, 199, 12), torch.randn(3, 80, 251)
        frames = torch.ones(3, 199, dtype=torch.bool); frames[0, 150:] = False
        mel_mask = torch.ones(3, 251, dtype=torch.bool); mel_mask[0, 190:] = False
        _, reference = recovery_loss(state, teacher, mel, decoder.normalizer, frames, mel_mask, torch.rand(3), torch.arange(3),
                                     duration_weight=0.)
        _, parts = um.aligned_loss(state, teacher, mel, decoder.normalizer, frames, mel_mask)
        for key in ('contrastive', 'sequence_mse', 'mel'):
            self.assertAlmostEqual(parts[key], reference[key], places=5)
        none = torch.zeros(3, 3, dtype=torch.bool)
        _, isolated = um.aligned_loss(state, teacher, mel, decoder.normalizer, frames, mel_mask, none)
        self.assertAlmostEqual(isolated['contrastive'], 0., places=6)
        teacher_norm = torch.randn(3, 20, 8)
        valid = um.music_negatives([1, 1, 2], [0., 2., 0.], teacher_norm)
        self.assertFalse(bool(valid[0, 1])); self.assertTrue(bool(valid[0, 2]))
        same_content = torch.cat([teacher_norm[:1], teacher_norm[:1], teacher_norm[2:]])
        self.assertFalse(bool(um.music_negatives([1, 2, 3], [0., 0., 0.], same_content)[0, 1]))


class MusicIOTests(unittest.TestCase):
    def cnd_structures(self):
        rng = np.random.default_rng(0)
        envelopes = [rng.normal(size=100), rng.normal(size=120)]
        features = np.empty((3, 4), object)
        for f in range(3):
            for trial, piece in enumerate([0, 1, 0, 1]):
                features[f, trial] = envelopes[piece][:, None] * (f + 1)
        names = np.empty((1, 3), object); names[0] = ['Envelope', 'Onsets', 'PitchOnsets']
        stim = dict(data=features, names=names, fs=64., stimIdxs=np.array([[1, 2, 1, 2]]))
        trials = np.empty((1, 4), object)
        for k in range(4):
            trials[0, k] = rng.normal(size=(300, 66))
        chanlocs = np.zeros((1, 66), dtype=[('labels', 'O'), ('X', 'O'), ('Y', 'O'), ('Z', 'O')])
        for i in range(66):
            chanlocs[0, i] = (f'A{i + 1}' if i < 32 else f'B{i - 31}' if i < 64 else f'EXG{i - 63}', 0., 0., 1.)
        eeg = dict(data=trials, fs=512., chanlocs=chanlocs, paddingStartSample=512., origTrialPosition=np.array([[3, 1, 4, 2]]))
        return stim, eeg, envelopes

    def test_cnd_v5_files(self):
        from scipy.io import savemat
        stim, eeg, envelopes = self.cnd_structures()
        with tempfile.TemporaryDirectory() as folder:
            savemat(Path(folder) / 'dataStim.mat', dict(stim=stim)); savemat(Path(folder) / 'dataSub1.mat', dict(eeg=eeg))
            stimulus = mio.cnd_stimulus(mio.load_mat(Path(folder) / 'dataStim.mat'))
            recording = mio.cnd_eeg(mio.load_mat(Path(folder) / 'dataSub1.mat'))
        self.assertEqual(stimulus.names, ['Envelope', 'Onsets', 'PitchOnsets']); self.assertEqual(stimulus.fs, 64.)
        self.assertEqual(stimulus.stimulus_index.tolist(), [1, 2, 1, 2])
        np.testing.assert_allclose(stimulus.features[1][3], 2 * envelopes[1])
        self.assertEqual(len(recording.trials), 4); self.assertEqual(recording.trials[0].shape, (300, 64))
        self.assertEqual(recording.labels[:2], ['A1', 'A2']); self.assertEqual(recording.labels[-1], 'B32')
        self.assertEqual(recording.padding_start, 512); self.assertEqual(recording.presentation.tolist(), [3, 1, 4, 2])
        self.assertEqual(mio.group_identical(stimulus.features[0]).tolist(), [1, 2, 1, 2])

    def test_cnd_v73_hdf5_files(self):
        """MATLAB v7.3 layout: transposed arrays, cells as object references, char as uint16."""
        _, _, envelopes = self.cnd_structures()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'dataStim.mat'
            with h5py.File(path, 'w', userblock_size=512) as h5:
                refs = h5.create_group('#refs#')
                stim = h5.create_group('stim'); stim.attrs['MATLAB_class'] = np.bytes_('struct')
                stim.create_dataset('fs', data=np.array([[64.]]))
                stim.create_dataset('stimIdxs', data=np.array([[1.], [2.], [1.], [2.]]))      # MATLAB 1x4
                data = stim.create_dataset('data', shape=(4, 3), dtype=h5py.ref_dtype)          # MATLAB 3x4
                for f in range(3):
                    for trial, piece in enumerate([0, 1, 0, 1]):
                        d = refs.create_dataset(f'd{f}{trial}', data=(envelopes[piece] * (f + 1))[None, :])   # MATLAB Nx1
                        data[trial, f] = d.ref
                names = stim.create_dataset('names', shape=(3, 1), dtype=h5py.ref_dtype)
                for f, text in enumerate(['Envelope', 'Onsets', 'PitchOnsets']):
                    d = refs.create_dataset(f'n{f}', data=np.array([[ord(c)] for c in text], np.uint16))
                    d.attrs['MATLAB_class'] = np.bytes_('char'); names[f, 0] = d.ref
            with open(path, 'r+b') as f:
                f.write(b'MATLAB 7.3 MAT-file, synthetic test'.ljust(116) + b'\x00' * 8 + b'\x00\x02IM')
            stimulus = mio.cnd_stimulus(mio.load_mat(path))
        self.assertEqual(stimulus.names, ['Envelope', 'Onsets', 'PitchOnsets']); self.assertEqual(stimulus.fs, 64.)
        self.assertEqual(stimulus.stimulus_index.tolist(), [1, 2, 1, 2])
        np.testing.assert_allclose(stimulus.features[2][1], 3 * envelopes[1])

    def test_midi_parser_tempo_map_running_status_and_render(self):
        def vlq(n):
            out = [n & 0x7F]; n >>= 7
            while n:
                out.insert(0, (n & 0x7F) | 0x80); n >>= 7
            return bytes(out)
        def track(events):
            body = b''.join(vlq(d) + e for d, e in events) + vlq(0) + b'\xFF\x2F\x00'
            return b'MTrk' + len(body).to_bytes(4, 'big') + body
        tempo = track([(0, b'\xFF\x51\x03' + (500000).to_bytes(3, 'big')), (960, b'\xFF\x51\x03' + (250000).to_bytes(3, 'big'))])
        notes = track([(0, b'\x90\x3C\x64'), (480, b'\x80\x3C\x00'), (480, b'\x90\x40\x50'), (480, b'\x40\x00')])  # running status
        data = b'MThd' + (6).to_bytes(4, 'big') + (1).to_bytes(2, 'big') + (2).to_bytes(2, 'big') + (480).to_bytes(2, 'big') + tempo + notes
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'x.mid'; path.write_bytes(data)
            parsed = mio.parse_midi(path)
        self.assertEqual([(n.pitch, round(n.start, 6), round(n.end, 6)) for n in parsed], [(60, 0., .5), (64, 1., 1.25)])
        audio = mio.render_additive(parsed, 24000)
        energy = lambda a, b: float(np.square(audio[int(a * 24000):int(b * 24000)]).mean())
        self.assertGreater(energy(1., 1.2), 50 * energy(.7, .95))

    def test_bigvgan_mel(self):
        basis = mio.mel_filterbank(24000, 1024, 100)
        self.assertEqual(basis.shape, (100, 513)); self.assertTrue((basis >= 0).all())
        self.assertTrue((np.diff(basis.argmax(1)) >= 0).all())
        tone = np.sin(2 * np.pi * 1000 * np.arange(96000) / 24000).astype(np.float32)
        mel = mio.bigvgan_mel(tone)
        self.assertEqual(mel.shape, (100, 375))
        centres = mio._mel_to_hz(np.linspace(mio._hz_to_mel(0), mio._hz_to_mel(12000), 102))[1:-1]
        self.assertLessEqual(abs(int(mel[:, 100:200].mean(1).argmax()) - int(np.abs(centres - 1000).argmin())), 1)
        self.assertAlmostEqual(float(mio.bigvgan_mel_times(375)[0]), 128 / 24000)


class MusinGPrepareTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(ROOT / 'scripts'))
        import prepare_musin_g
        self.prep = prepare_musin_g

    def test_egi_net_is_trimmed_to_the_biosemi_coverage_on_one_sphere(self):
        import mne
        names = mne.channels.make_standard_montage('GSN-HydroCel-129').ch_names[:-1] + ['E129']
        xyz = self.prep.montage_positions(names)
        np.testing.assert_allclose(np.linalg.norm(xyz, axis=1), .095, atol=1e-9)
        kept = self.prep.coverage_distance(xyz) <= 15.
        self.assertEqual(int(kept.sum()), 109)
        self.assertTrue(kept[names.index('E129')])                                   # Cz (the reference) stays
        self.assertFalse(any(kept[names.index(n)] for n in ('E125', 'E126', 'E127', 'E128')))   # eye electrodes go

    def test_reference_channel_is_never_flagged_and_ratings_parse(self):
        rng = np.random.default_rng(0)
        data = rng.normal(scale=1e-5, size=(8, 2500)); data[3] = 0.; data[5] *= 40
        self.assertEqual(mio.bad_channels(data, 250., exclude=[3]).tolist(), [5])
        self.assertEqual(sorted(mio.bad_channels(data, 250.).tolist()), [3, 5])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'Behavioural_data'
            path.write_text('Subject\t\tSong_ID\t\tEnjoyment\tFamiliarity\n1\t\t1\t\t2\t\t3\n20\t\t12\t\t5\t\t1\n')
            self.assertEqual(self.prep.behaviour(path), {(1, 1): (2, 3), (20, 12): (5, 1)})

    def test_splits_do_not_depend_on_what_is_downloaded(self):
        held = sorted(mio.ranked(self.prep.PARTICIPANTS, 'musin_g-subject')[:4])
        self.assertEqual(held, [2, 6, 7, 14])
        self.assertEqual(mio.ranked(list(range(1, 13)), 'musin_g-song')[:2], [11, 1])


class MusicWindowTests(unittest.TestCase):
    def test_windows_are_aligned_to_the_stimulus_clock(self):
        with tempfile.TemporaryDirectory() as folder:
            manifest, targets, normalizer = make_music(folder)
            windows = md.MusicWindows('/', manifest, targets, normalizer)
            rng = np.random.default_rng(0)
            self.assertTrue(set(windows.frame.piece) == {1, 2} and set(windows.frame.subject) == {'sub-01', 'sub-02'})
            self.assertAlmostEqual(float(windows.limits.max()), 8.)
            keys = windows.random_keys(3, rng)
            batch = windows.load(keys)
            self.assertEqual(tuple(batch['eeg'].shape), (3, 8, EEG_SAMPLES))
            for i, (row, start) in enumerate(keys):
                piece = int(windows.frame.piece.iat[row])
                first = float(batch['eeg'][i, 2, 0]) - 200 - 1000 * piece
                self.assertAlmostEqual(first, start + EEG_START, delta=1 / 256 + 1e-4)
                np.testing.assert_allclose(batch['teacher'][i, :, 0].numpy(), start + windows.targets.teacher_grid, atol=1e-4)
                np.testing.assert_allclose(batch['mel'][i, 0].numpy(), start + windows.targets.mel_grid, atol=1e-4)
                np.testing.assert_allclose(batch['acoustic'][i, 0].numpy(), start + np.arange(256) / 64, atol=1e-4)
            by_piece = {}
            for row, start in keys:
                by_piece.setdefault(int(windows.frame.piece.iat[row]), []).append(start)
            for starts in by_piece.values():
                self.assertTrue(all(abs(a - b) >= 4 for i, a in enumerate(starts) for b in starts[i + 1:]))
            partners = windows.partners(keys[0], 5, rng)
            self.assertTrue(partners and all(p[1] == keys[0][1] and p[0] != keys[0][0] and
                                             windows.frame.piece.iat[p[0]] == windows.frame.piece.iat[keys[0][0]] for p in partners))
            other = windows.background(keys[0], rng)
            self.assertEqual(windows.frame.subject.iat[other[0]], windows.frame.subject.iat[keys[0][0]])
            self.assertNotEqual(windows.frame.piece.iat[other[0]], windows.frame.piece.iat[keys[0][0]])
            for start in (0., 1., 4., 8.):
                shifted = windows.shifted((0, start))
                self.assertTrue(shifted[0] != 0 or abs(shifted[1] - start) >= 4, (start, shifted))
            single = md.MusicWindows('/', manifest, targets, normalizer, content_roles=('validation',))
            self.assertNotEqual(single.shifted((0, 1.)), (0, 1.))                 # one piece only: same trial, another moment
            single.close()
            self.assertTrue(all(0 <= s <= windows.limits[r] for r, s in windows.grid_keys()))
            eeg, masks, groups = windows.eeg_bank_sample(2, rng)
            self.assertEqual(eeg.shape[0], 4); self.assertEqual(sorted(set(groups)), ['toy:sub-01', 'toy:sub-02'])
            windows.close()


class TrainingTests(unittest.TestCase):
    def arguments(self, output, music_targets, music_decoder, updates=4):
        p = ut.parser()
        args = p.parse_args([
            '--output', str(output), '--device', 'cpu', '--updates', str(updates), '--eval-every', '2', '--batch-size', '3',
            '--music-batch', '3', '--virtual', '8', '--width', '16', '--harmonics', '4', '--warmup', '1', '--mel-warmup', '2',
            '--shift-max', '4', '--channel-gain', '.1', '--background-mix', '.5', '--mix-same-content', '.5', '--mix-partners', '2',
            '--mix-start', '1', '--mix-anneal-epochs', '2', '--spatial-geometry', '.5', '--spatial-subset', '.5',
            '--spatial-reference', '.5', '--spatial-conduction', '.5', '--spatial-recolour', '.5', '--consistency-weight', '.1',
            '--music-targets', str(music_targets), '--music-decoder', str(music_decoder), '--music-mix', '.5',
            '--music-background', '.5', '--music-eval-windows', '6'])
        ut.check_args(p, args)
        return args

    def test_joint_training_evaluates_checkpoints_and_resumes_exactly(self):
        train = FakeSpeech(['s1', 's2', 's3'], ['a', 'b', 'c', 'd'])
        validation = FakeSpeech(['s1', 's2', 's4', 's5'], ['e', 'f'], seed=1)
        cohorts = {'seen': ut.subset(validation, [0, 1, 2, 3]), 'unseen': ut.subset(validation, [4, 5, 6, 7])}
        speech_payload = decoder_payload(tiny_decoder())
        music_payload = decoder_payload(tiny_decoder(8, 100, 299, 375))
        with tempfile.TemporaryDirectory() as folder:
            manifest, targets, normalizer = make_music(Path(folder) / 'music')
            def run(output, **patches):
                music_train = md.MusicWindows('/', manifest, targets, normalizer)
                music_cohorts = {'seen': md.MusicWindows('/', manifest, targets, normalizer, content_roles=('validation',)),
                                 'unseen': md.MusicWindows('/', manifest, targets, normalizer, content_roles=('validation',),
                                                           subject_roles=('heldout',))}
                ut.train(self.arguments(output, targets, 'unused.pt'), speech_train=train, speech_full_train=train,
                         speech_cohorts=cohorts, templates=templates(), speech_decoder_payload=speech_payload,
                         signature_extra=dict(heldout_subjects=['s4', 's5']), music_train=music_train,
                         music_cohorts=music_cohorts, music_decoder_payload=music_payload, bank_factory=FakeBank)
            base = Path(folder)
            run(base / 'continuous')
            metrics = json.loads((base / 'continuous/metrics.json').read_text())
            self.assertEqual([h['update'] for h in metrics['history']], [2, 4])
            row = metrics['history'][-1]
            self.assertEqual(row['selection'], 'unseen'); self.assertEqual(set(row['speech']), {'seen', 'unseen'})
            self.assertEqual(set(row['music']), {'seen', 'unseen'})
            self.assertTrue(np.isfinite(row['music']['unseen']['retrieval_mrr']))
            model, payload = ut.load_universal(base / 'continuous/best_metric.pt')
            self.assertEqual(payload['heldout_subjects'], ['s4', 's5'])
            self.assertFalse(any('subject' in n for n, _ in model.named_parameters()))
            original = torch.optim.AdamW.step; calls = [0]
            def interrupt_after_second(optimizer, *a, **kw):
                result = original(optimizer, *a, **kw); calls[0] += 1
                if calls[0] == 2:
                    os.kill(os.getpid(), signal.SIGINT)
                return result
            with patch.object(torch.optim.AdamW, 'step', interrupt_after_second):
                with self.assertRaises(SystemExit):
                    run(base / 'resumed')
            self.assertEqual(torch.load(base / 'resumed/training_state.pt', weights_only=False)['step'], 2)
            run(base / 'resumed')
            final = torch.load(base / 'continuous/training_state.pt', weights_only=False)['model']
            resumed = torch.load(base / 'resumed/training_state.pt', weights_only=False)['model']
            for key in final:
                torch.testing.assert_close(resumed[key], final[key], msg=key)


if __name__ == '__main__':
    unittest.main()
