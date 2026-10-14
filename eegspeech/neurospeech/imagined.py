"""Imagined speech as generation, transferred from the listened-speech model.

Nothing here classifies. Imagined EEG goes through the frozen listened-speech
encoder; a small re-timing adapter, the only trained part, turns its sequence into
the continuous condition the frozen flow generator already understands, and the
generator renders a waveform. Imagination has no audio of its own, so the
regression target is the same trial's spoken recording. Prompt identity is never
an input or an output; it appears only in the measures and the split.
"""
from __future__ import annotations
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, default_collate
from ..utils import atomic_checkpoint, atomic_json, append_jsonl, load_checkpoint, read_json, seed_all, stable_int
from .data import auditory_filter
from .models import time_encoding

PROMPTS = ('iy', 'uw', 'piy', 'tiy', 'diy', 'm', 'n', 'pat', 'pot', 'knew', 'gnaw')
CONTROLS = ('imagined', 'wrong-prompt', 'rest', 'null')


def interface(cfg):
    """State-bounded window: the whole EEG context has to lie inside the 5 s imagined state."""
    p = cfg['imagined']; context = p['left']+p['core_seconds']+p['right']
    return {'left': p['left'], 'core': p['core_seconds'], 'samples': round(context*200), 'frames': round(p['core_seconds']*50),
            'mel_frames': round(p['core_seconds']*24000)//256+1, 'context': context}


def trial_rows(cfg):
    """One row per KaraOne trial with an imagined record long enough for the interface."""
    folder = cfg.dir('processed')/'karaone'; manifest = read_json(folder/'manifest.json'); need = interface(cfg)['context']; trials = defaultdict(dict)
    for record in manifest['records'].values(): trials[(record['subject'], int(record['metadata']['trial']))][record['task']] = record
    rows = []
    for (subject, trial), states in sorted(trials.items()):
        record = states.get('imagined'); rest = states.get('rest')
        if record is None or record['duration'] < need or record['audio'] not in manifest['contents']: continue
        usable_rest = rest is not None and rest['duration'] >= need
        rows.append({'dataset': 'karaone', 'subject': subject, 'trial': trial, 'prompt': record['metadata']['prompt'], 'id': f'karaone:{subject}:{trial:04d}',
                     'path': record['path'], 'offset': (record['duration']-need)/2, 'good_channels': record['good_channels'],
                     'rest_path': rest['path'] if usable_rest else None, 'rest_offset': (rest['duration']-need)/2 if usable_rest else None,
                     'audio': record['audio'], 'audio_path': manifest['contents'][record['audio']]['path'], 'audio_sha256': manifest['contents'][record['audio']]['sha256']})
    return rows


def split_rows(rows, mode='subject', held=0, seed=20261013):
    """Fixed by metadata. ``subject``: unseen people, same prompts. ``prompt``: one prompt never used for transfer."""
    people = sorted({r['subject'] for r in rows}, key=lambda s: stable_int(f'imagined:{seed}:{s}'))
    development, others = set(people[:2]), people[2:]
    if mode == 'subject':
        test = set(others[int(held)::4])
        if not test: raise ValueError('Subject folds are 0..3')
        pick = lambda r: 'validation' if r['subject'] in development else 'test' if r['subject'] in test else 'train'
    elif mode == 'prompt':
        if held not in PROMPTS: raise ValueError(f'Held-out prompt must be one of {PROMPTS}')
        pick = lambda r: 'test' if r['prompt'] == held else 'validation' if r['subject'] in development else 'train'
    else: raise ValueError(mode)
    return {name: [r for r in rows if pick(r) == name] for name in ('train', 'validation', 'test')}


def wrong_prompt(rows):
    """Deterministic other-prompt trial of the same person, the control for 'any imagery produces this sound'."""
    by = defaultdict(list)
    for i, r in enumerate(rows): by[r['subject']].append(i)
    mapping = {}
    for i, r in enumerate(rows):
        candidates = [j for j in by[r['subject']] if rows[j]['prompt'] != r['prompt']]
        if candidates: mapping[i] = min(candidates, key=lambda j: stable_int(f"imagined-wrong:{r['id']}:{rows[j]['id']}"))
    return mapping


class SpokenTargets:
    """Speech-model targets of a trial's spoken recording on the imagined time grid, cached per waveform."""
    def __init__(self, cfg, audio):
        from .neural import AcousticFrontend
        self.cfg, self.audio, self.frontend, self.memory = cfg, audio, AcousticFrontend(), {}
        self.folder = cfg.dir('artifacts')/'imagined'/'target_cache'

    def wave(self, row):
        """First ``core`` seconds of the recording, zero-padded: prompts are shorter than the window."""
        core = round(interface(self.cfg)['core']*24000); wave, rate = sf.read(self.cfg.dir('processed')/'karaone'/row['audio_path'], dtype='float32')
        if rate != 24000: raise ValueError('Processed audio must be 24 kHz')
        return np.pad(wave[:core], (0, max(0, core-len(wave))))

    def features(self, wave):
        """[frames,768] teacher, [100,mel_frames] Mel and [frames,2] rhythm of one core waveform; also used on generated audio."""
        from scipy.signal import resample_poly
        face = interface(self.cfg); times = (np.arange(face['frames'])+.5)/50; second = np.zeros(24000, np.float32)
        feature, t = self.audio.wavlm(resample_poly(np.concatenate((second, wave, second)), 2, 3).astype(np.float32), crop_start=-1, core_seconds=face['core'])
        feature = feature[0].cpu().numpy(); t = t.cpu().numpy()
        teacher = np.stack([np.interp(times, t, feature[:, i]) for i in range(feature.shape[1])], 1).astype(np.float32)
        with torch.no_grad(): mel = self.audio.vocos.features(torch.from_numpy(wave)[None].to(self.audio.device))[0].cpu().numpy()
        context = np.concatenate((second, wave, np.zeros(round(self.cfg['imagined']['right']*24000), np.float32)))
        phi64 = self.frontend.numpy(context)[0][64:64+round(face['core']*64)][:, [0, 9]]
        phi = np.stack([np.interp(times, np.arange(len(phi64))/64, phi64[:, i]) for i in range(2)], 1).astype(np.float32)
        return {'teacher': teacher, 'mel': mel.astype(np.float32), 'phi': phi}

    def get(self, row):
        key = row['audio_sha256']
        if key not in self.memory:
            path = self.folder/f'{key}.npz'
            if path.exists():
                with np.load(path, allow_pickle=False) as z: self.memory[key] = {k: z[k].copy() for k in z.files}
            else:
                self.memory[key] = self.features(self.wave(row)); self.folder.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(path.with_suffix('.part.npz'), **self.memory[key]); path.with_suffix('.part.npz').replace(path)
        return self.memory[key]


class ImaginedDataset(Dataset):
    """Imagined (or rest) EEG of a trial in the listened-speech input format, normalized with listened-speech statistics."""
    def __init__(self, cfg, rows, stats, targets=None, state='imagined'):
        if state not in ('imagined', 'rest'): raise ValueError(state)
        self.cfg, self.stats, self.targets, self.state, self.face = cfg, stats, targets, state, interface(cfg)
        self.rows = [r for r in rows if state == 'imagined' or r['rest_path']]; self.meta = {}

    def __len__(self): return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]; face = self.face; folder = self.cfg.dir('processed')/'karaone'
        record = folder/(r['path'] if self.state == 'imagined' else r['rest_path']); offset = r['offset'] if self.state == 'imagined' else r['rest_offset']
        if str(record) not in self.meta: self.meta[str(record)] = read_json(record/'meta.json')
        meta = self.meta[str(record)]; x = np.load(record/'eeg.npy', mmap_mode='r'); a = round(offset*200)
        eeg = np.array(x[:, a:a+face['samples']].T, dtype=np.float32); auditory = auditory_filter(np.array(x.T))[a:a+face['samples']]
        if eeg.shape[0] != face['samples']: raise ValueError('Imagined state is shorter than the transfer window')
        good = np.asarray(meta['good_channels'], bool)&(eeg.std(0) >= self.cfg['preprocessing']['flat_uv'])&(np.ptp(eeg, axis=0) <= self.cfg['preprocessing']['peak_to_peak_uv'])
        scale = lambda v, kind: (v-np.asarray(self.stats[kind]['mean'], np.float32))/np.asarray(self.stats[kind]['std'], np.float32)
        eeg = scale(eeg, 'eeg'); auditory = scale(auditory, 'auditory'); eeg[:, ~good] = 0; auditory[:, ~good] = 0
        sample = {'eeg': torch.tensor(eeg), 'auditory': torch.tensor(auditory), 'positions': torch.tensor(meta['positions'], dtype=torch.float32),
                  'channel_mask': torch.tensor(good), 'times': torch.tensor((np.arange(face['frames'])+.5)/50, dtype=torch.float32),
                  'eeg_times': torch.tensor(np.arange(face['samples'])/200-face['left'], dtype=torch.float32),
                  'frame_mask': torch.ones(face['frames'], dtype=torch.bool), 'subject': r['subject'], 'prompt': r['prompt'], 'id': r['id'], 'state': self.state}
        if self.targets is not None:
            for key, value in self.targets.get(r).items(): sample[key] = torch.tensor(scale(value.T, key).T if key == 'mel' else scale(value, key))
            sample['wave'] = torch.tensor(self.targets.wave(r))
        return sample


class Retimer(nn.Module):
    """Re-times the listened-speech representation for imagery, whose timing inside the 5 s is unknown.

    Every output frame attends over the whole EEG sequence. The update enters through a
    zero-initialised projection, so before any transfer training the model is exactly the
    listened-speech model applied to imagined EEG.
    """
    def __init__(self, width=256, heads=4, layers=2):
        super().__init__(); self.width = width
        self.attention = nn.ModuleList([nn.MultiheadAttention(width, heads, batch_first=True) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(width) for _ in range(layers)])
        self.out = nn.Linear(width, width); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, hidden):
        times = (torch.arange(hidden.shape[1], device=hidden.device)+.5)/50
        h = keys = hidden+time_encoding(times, self.width).to(hidden)
        for attention, norm in zip(self.attention, self.norm): h = h+attention(norm(h), keys, keys, need_weights=False)[0]
        return hidden+self.out(h)


def transfer_loss(output, batch):
    """Continuous regression onto the spoken recording's speech-model features; no class term exists."""
    mask = batch['frame_mask'][..., None].to(output['teacher'].dtype); count = mask.sum().clamp_min(1)
    feature = (F.smooth_l1_loss(output['teacher'], batch['teacher'], reduction='none').mean(-1, keepdim=True)*mask).sum()/count
    cosine = ((1-F.cosine_similarity(output['teacher'], batch['teacher'], dim=-1))[..., None]*mask).sum()/count
    rhythm = ((output['phi']-batch['phi']).square().mean(-1, keepdim=True)*mask).sum()/count
    return feature+.5*cosine+.2*rhythm


def transfer_dir(cfg, protocol, fold, seed, system, mode, held):
    return cfg.dir('artifacts')/'imagined'/f'{protocol}-{fold}-{seed}-{system}'/f'{mode}-{held}'


def _batch(sample, device):
    from .training import move_batch
    return move_batch(default_collate([sample]), device)


def _condition(generator, batch, retimer=None, null=False):
    """Condition for the frozen generator. Reliability stays neutral: the listened-speech calibrator is unverified here."""
    from .inference import condition_from_output, eeg_only
    hidden = generator.encoder.hidden(eeg_only(batch)); output = generator.encoder.heads(hidden if retimer is None else retimer(hidden))
    if generator.system in ('S0', 'S1'): output['phi'] = output['phi']*0
    output['times'] = batch['times']; condition = condition_from_output(output, None, generator.system)
    if null: condition = {**condition, **{k: torch.zeros_like(condition[k]) for k in ('content', 'rhythm', 'reliability')}}
    return output, condition


def load_listened(cfg, protocol, fold, seed, system, device):
    """The frozen listened-speech generator this transfer starts from; it must already be trained."""
    from .inference import load_generator
    from ..perceived_audio import FrozenAudio
    audio = FrozenAudio(cfg, device)
    return load_generator(cfg, protocol, fold, seed, system, device, audio.vocos), audio


def train_transfer(cfg, protocol='P1', fold='F0', seed=11, system='S2', mode='subject', held=0, device='cpu', generator=None, audio=None, rows=None):
    """Train only the re-timer on imagined EEG; encoder, heads and generator stay frozen."""
    p = cfg['imagined']; seed_all(seed)
    if generator is None: generator, audio = load_listened(cfg, protocol, fold, seed, system, device)
    generator.encoder.eval().requires_grad_(False); split = split_rows(trial_rows(cfg) if rows is None else rows, mode, held); targets = SpokenTargets(cfg, audio)
    train = ImaginedDataset(cfg, split['train'], generator.stats, targets); validation = ImaginedDataset(cfg, split['validation'], generator.stats, targets)
    if not len(train) or not len(validation): raise ValueError('Empty imagined transfer split')
    retimer = Retimer().to(device); optimizer = torch.optim.AdamW(retimer.parameters(), lr=p['lr'], weight_decay=.01); rng = np.random.default_rng(seed)
    folder = transfer_dir(cfg, protocol, fold, seed, system, mode, held); folder.mkdir(parents=True, exist_ok=True); other = wrong_prompt(validation.rows); best = float('inf')
    for step in range(p['steps']):
        retimer.train(); optimizer.zero_grad(set_to_none=True); total = 0.
        for index in rng.integers(len(train), size=p['batch']):
            batch = _batch(train[int(index)], device)
            with torch.no_grad(): hidden = generator.encoder.hidden({k: batch[k] for k in ('eeg', 'auditory', 'positions', 'channel_mask', 'times')})
            loss = transfer_loss(generator.encoder.heads(retimer(hidden)), batch)/p['batch']; loss.backward(); total += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(retimer.parameters(), 1, error_if_nonfinite=True); optimizer.step()
        if (step+1) % 50 == 0: append_jsonl(folder/'training.jsonl', {'step': step+1, 'loss': total})
        if (step+1) % p['validate_every'] == 0 or step+1 == p['steps']:
            retimer.eval(); errors = defaultdict(list)
            with torch.no_grad():
                for i in range(len(validation)):
                    batch = _batch(validation[i], device); own = float(transfer_loss(_condition(generator, batch, retimer)[0], batch))
                    if i in other: errors[validation.rows[i]['subject']].append((own, float(transfer_loss(_condition(generator, _batch(validation[other[i]], device), retimer)[0], batch))))
            score = float(np.mean([np.mean([a for a, _ in v]) for v in errors.values()])); wrong = float(np.mean([np.mean([b for _, b in v]) for v in errors.values()]))
            saved = {'retimer': retimer.state_dict(), 'step': step+1, 'validation': {'own_trial_loss': score, 'wrong_prompt_loss': wrong},
                     'transfer': {'from': [protocol, fold, seed, system], 'mode': mode, 'held': held, 'config_hash': cfg.hash,
                                  'trained': 'retimer only; listened-speech encoder, heads and flow frozen', 'train_trials': len(train)}}
            if score < best: best = score; atomic_checkpoint(folder/'best.pt', saved)
            atomic_checkpoint(folder/'latest.pt', saved)
    result = load_checkpoint(folder/'best.pt', device); atomic_json(folder/'validation.json', {**result['validation'], 'selected_step': result['step']})
    return result['validation']


def _generate(generator, condition, frames, key, device):
    from .inference import native_decode
    noise = torch.randn(1, 100, frames, generator=torch.Generator(device=device).manual_seed(stable_int(key)), device=device)
    mel = generator.flow.rollout(noise, condition, steps=32, method='heun')
    return mel, native_decode(mel, generator.stats, generator.vocos)[0]


@torch.no_grad()
def evaluate_transfer(cfg, protocol='P1', fold='F0', seed=11, system='S2', mode='subject', held=0, adapted=True, device='cpu',
                      generator=None, audio=None, rows=None, retimer=None, limit=None):
    """Generate audio from held-out imagined EEG and from three controls; every number is measured on generated audio.

    ``adapted=False`` is the plain listened-speech model on imagined EEG (zero-shot).
    The prompt rank compares generated audio with spoken recordings of other people; it is a
    measure, not a step of the generation path.
    """
    if generator is None: generator, audio = load_listened(cfg, protocol, fold, seed, system, device)
    generator.encoder.eval(); generator.flow.eval(); face = interface(cfg); folder = transfer_dir(cfg, protocol, fold, seed, system, mode, held)
    if adapted and retimer is None:
        retimer = Retimer().to(device); retimer.load_state_dict(load_checkpoint(folder/'best.pt', device)['retimer'])
    if retimer is not None: retimer.eval()
    everything = trial_rows(cfg) if rows is None else rows; split = split_rows(everything, mode, held); targets = SpokenTargets(cfg, audio)
    test = ImaginedDataset(cfg, split['test'][:limit], generator.stats, targets); rest = ImaginedDataset(cfg, split['test'][:limit], generator.stats, targets, 'rest')
    rest_index = {r['id']: i for i, r in enumerate(rest.rows)}; other = wrong_prompt(test.rows)
    scale = lambda v, kind: (v-np.asarray(generator.stats[kind]['mean'], np.float32))/np.asarray(generator.stats[kind]['std'], np.float32)
    # Reference of each prompt for one person: mean spoken-recording features of everybody else.
    spoken = defaultdict(list)
    for r in everything: spoken[(r['subject'], r['prompt'])].append(scale(targets.get(r)['teacher'], 'teacher'))
    def references(subject): return {q: np.mean([f for (s, p), v in spoken.items() if s != subject and p == q for f in v], 0) for q in PROMPTS}
    unit = lambda v: v/np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-8)
    name = 'transfer' if adapted else 'zero-shot'; out = folder/'evaluation'/name; out.mkdir(parents=True, exist_ok=True)
    if (out/'scores.jsonl').exists(): raise FileExistsError('Scores already exist; use a fresh evaluation namespace')
    results = []; heard = {}; cache = {}
    for i in range(len(test)):
        r = test.rows[i]; batch = _batch(test[i], device); subject = r['subject']
        if subject not in cache: cache = {subject: references(subject)}
        sources = {'imagined': batch, 'null': batch}
        if i in other: sources['wrong-prompt'] = _batch(test[other[i]], device)
        if r['id'] in rest_index: sources['rest'] = _batch(rest[rest_index[r['id']]], device)
        for control in CONTROLS:
            if control not in sources: continue
            _, condition = _condition(generator, sources[control], retimer, null=control == 'null')
            mel, wave = _generate(generator, condition, face['mel_frames'], f"imagined-noise:{r['id']}", device); wave = wave.cpu().numpy()
            made = unit(scale(targets.features(wave[:round(face['core']*24000)].astype(np.float32))['teacher'], 'teacher')); own = unit(batch['teacher'][0].cpu().numpy())
            similarity = {q: float((made*unit(v)).sum(-1).mean()) for q, v in cache[subject].items()}
            order = sorted(similarity, key=similarity.get, reverse=True)
            row = {'id': r['id'], 'subject': subject, 'prompt': r['prompt'], 'control': control, 'system': name,
                   'similarity_to_own_recording': float((made*own).sum(-1).mean()), 'mel_mae': float((mel-batch['mel']).abs().mean()),
                   'prompt_rank': order.index(r['prompt'])+1, 'prompts': len(order)}
            results.append(row); append_jsonl(out/'scores.jsonl', row)
            if heard.setdefault((subject, r['prompt']), r['id']) == r['id']:
                # Listening set fixed by metadata: the first held-out trial of each person and prompt, every control.
                stem = out/f"{subject}_{r['prompt']}_{r['trial']:04d}"
                if control == 'imagined': sf.write(f'{stem}_spoken.wav', batch['wave'][0].cpu().numpy(), 24000)
                sf.write(f'{stem}_{control}.wav', wave, 24000)
    from .reporting import stratified_paired
    people = defaultdict(lambda: defaultdict(list))
    for row in results: people[row['control']][row['subject']].append(row)
    mean = lambda control, key: float(np.mean([np.mean([x[key] for x in v]) for v in people[control].values()])) if people[control] else None
    summary = {'system': name, 'mode': mode, 'held': held, 'trials': len(test), 'chance_top1': 1/len(PROMPTS), 'controls': {
        c: {'similarity_to_own_recording': mean(c, 'similarity_to_own_recording'), 'mel_mae': mean(c, 'mel_mae'), 'mean_prompt_rank': mean(c, 'prompt_rank'),
            'prompt_top1': float(np.mean([np.mean([x['prompt_rank'] == 1 for x in v]) for v in people[c].values()])) if people[c] else None} for c in CONTROLS}}
    for control in ('wrong-prompt', 'rest', 'null'):
        shared = [s for s in people['imagined'] if s in people[control]]
        if len(shared) >= 2:
            delta = [np.mean([x['similarity_to_own_recording'] for x in people['imagined'][s]])-np.mean([x['similarity_to_own_recording'] for x in people[control][s]]) for s in shared]
            summary[f'imagined_minus_{control}'] = stratified_paired({'karaone': delta})
    atomic_json(out/'summary.json', summary)
    return summary
