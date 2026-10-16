"""Where does the gap to Défossez et al. (2023) come from? The same two-tower alignment, run under switchable protocols.

Three switches, one dataset per model (as in the paper):

* split: 'sentence' (the paper: sentences are assigned to train / validation / test, 70 / 20 / 10 %, identical
  sentences always to the same split, and 3 s windows that touch two splits are dropped) or 'content' (ours: whole
  content blocks, from the registry);
* subjects: 'all' (the paper: every participant is trained on, and tested on held-out sentences) or 'train' (ours:
  the P1 F0 training participants only; the others are reported separately as never seen);
* the evaluation is always the paper's: one candidate per test word, a 3 s window from 0.5 s before its onset
  (neighbouring candidates overlap), top-1 / top-10 among all test candidates of the dataset.

Speech features stay those of the content-block cache: with a sentence split their context may cross a sentence
boundary, as with features computed on whole recordings; this is part of what the sentence protocol allows.
"""
from __future__ import annotations
from collections import defaultdict
import hashlib
import math
import re
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from ..utils import atomic_json, append_jsonl, read_json
from .softalign import (SEG, FRAME, Aligner, Cache, alignment_losses, covers, eeg_window, identification, lr_at, status, to)
from .reconstruct import word_spans

SPLITS = ('train', 'validation', 'test')
PAPER = {'ds004408': {'name': 'Broderick2019', 'candidates': 1842, 'top1': .050, 'top10': .177},
         'brennan2019': {'name': 'Brennan2019', 'candidates': 190, 'top1': .052, 'top10': .257}}


def sentence_split(text, fractions=(.7, .2, .1)):
    """Split of a sentence from a hash of its normalised text, so identical sentences share a split."""
    key = re.sub(r'[^a-z ]', '', text.lower()).strip(); u = int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)/16**8
    return SPLITS[0] if u < fractions[0] else SPLITS[1] if u < fractions[0]+fractions[1] else SPLITS[2]


def sentences_of(spans, gap=.4):
    """Word index ranges of sentence-like stretches: break at final punctuation or at pauses >= gap seconds."""
    starts, ends, texts = spans; out, current = [], []
    for k in range(len(starts)):
        current.append(k)
        if k == len(starts)-1 or texts[k].rstrip().endswith(('.', '?', '!')) or starts[k+1]-ends[k] >= gap: out.append(current); current = []
    return out


class Plan:
    """For one dataset: which 3 s windows belong to which split, on the 40 ms grid and at word onsets - 0.5 s."""
    def __init__(self, cfg, cache, dataset, split='sentence', offset=-.5):
        self.dataset, self.split = dataset, split; spans = word_spans(cfg, cache); rows = cache.records[dataset]
        self.grid, self.words, self.summary = defaultdict(dict), defaultdict(dict), defaultdict(float)
        for content in sorted({r['audio'] for r in rows}):
            starts, ends, texts = spans[dataset, content]
            if split == 'sentence':
                label = np.empty(len(starts), object)
                for s in sentences_of(spans[dataset, content]):
                    which = sentence_split(' '.join(texts[k] for k in s)); label[s] = which; self.summary[which] += ends[s[-1]]-starts[s[0]]
            else:
                blocks = cache.spans[dataset][content]
                label = np.array([next((b[2] for b in blocks if b[0] <= (s+e)/2 < b[1]), None) for s, e in zip(starts, ends)], object)
                for b in blocks: self.summary[b[2]] += b[1]-b[0]
            last = max(ends) if len(ends) else 0.

            def assign(anchors):
                out = {k: [] for k in SPLITS}
                for a in anchors:
                    inside = np.flatnonzero((starts < a+SEG*FRAME) & (ends > a)); kinds = set(label[inside])
                    if len(inside) and len(kinds) == 1 and None not in kinds: out[kinds.pop()].append(a)
                return {k: np.array(v) for k, v in out.items()}
            grid = assign([round(a/FRAME)*FRAME for a in np.arange(0, max(0., last-SEG*FRAME), FRAME)])
            words = assign(sorted({round((s+offset)/FRAME)*FRAME for s in starts if s+offset >= 0}))
            for k in SPLITS: self.grid[k][content] = grid[k]; self.words[k][content] = words[k]

    def items(self, cache, split, people):
        """Evaluation windows: every record of `people` at every word-anchored window of `split` it covers."""
        rows = cache.records[self.dataset]
        return [(self.dataset, i, float(a)) for i, r in enumerate(rows) if r['key'] in people and r.get('condition', 'clean') == 'clean'
                for a in self.words[split].get(r['audio'], ()) if covers(r, a)]


class PlanStream(IterableDataset):
    """Training batches of M anchors from the plan's training windows, each seen through up to S people."""
    def __init__(self, cache, plan, subjects, anchors=32, views=4, seed=11):
        self.cache, self.plan, self.anchors, self.views, self.seed = cache, plan, anchors, views, seed
        rows = cache.records[plan.dataset]; self.by_audio = defaultdict(list)
        for i, r in enumerate(rows):
            if r['key'] in subjects and r.get('condition', 'clean') == 'clean': self.by_audio[r['audio']].append(i)
        self.contents = [c for c in sorted(self.by_audio) if len(plan.grid['train'].get(c, ()))]
        weight = np.array([len(plan.grid['train'][c]) for c in self.contents], float); self.weight = weight/weight.sum()

    def batch(self, rng):
        dataset = self.plan.dataset; rows = self.cache.records[dataset]; chosen = []; tries = 0
        while len(chosen) < self.anchors and tries < self.anchors*30:
            tries += 1; content = self.contents[rng.choice(len(self.contents), p=self.weight)]; grid = self.plan.grid['train'][content]
            a = float(grid[rng.integers(len(grid))])
            if any(c == content and abs(a-b) < SEG*FRAME for c, b, _ in chosen): continue
            people = defaultdict(list)
            for i in self.by_audio[content]:
                if covers(rows[i], a): people[rows[i]['key']].append(i)
            order = list(people); rng.shuffle(order); picked = []
            while len(picked) < self.views and any(people.values()):
                for key in order:
                    if people[key] and len(picked) < self.views: picked.append(people[key].pop(int(rng.integers(len(people[key])))))
            if picked: chosen.append((content, a, picked))
        eeg, mask, pos, subject, anchor = [], [], [], [], []
        for m, (_, a, picked) in enumerate(chosen):
            for i in picked:
                w, ok = eeg_window(rows[i], a, self.cache.good[dataset][i], True, rng)
                eeg.append(w); mask.append(ok); pos.append(self.cache.positions[dataset][i]); subject.append(self.cache.subject_id(rows[i]['key'])); anchor.append(m)
        return {'eeg': torch.from_numpy(np.stack(eeg)), 'mask': torch.from_numpy(np.stack(mask)), 'positions': torch.from_numpy(np.stack(pos)),
                'subject': torch.tensor(subject), 'anchor': torch.tensor(anchor), 'dataset': dataset, 'anchors': [(dataset, c, a) for c, a, _ in chosen]}

    def __iter__(self):
        info = get_worker_info(); worker = info.id if info else 0; rng = np.random.default_rng(self.seed+7919*worker+int(time.time()) % 100000)
        while True: yield self.batch(rng)


def setup(cfg, cache_dir, dataset, split, subjects):
    """Cache restricted to one dataset, the subject vocabulary of the chosen protocol, and the split plan."""
    registry = cfg.registry(); everyone = sorted(registry['subjects'][dataset])
    trained = everyone if subjects == 'all' else [s for s in cfg.fold_subjects('P1', 'F0', 'train') if s.startswith(dataset+':')]
    cache = Cache(cache_dir, trained); cache.records = {dataset: cache.records[dataset]}
    return cache, Plan(cfg, cache, dataset, split), everyone


def train(cfg, cache_dir, folder, dataset, split, subjects, deadline, device='cpu', threads=6, workers=2, seed=11, max_steps=None, validate_minutes=5, limit=400):
    """Main alignment variant, one dataset, until `deadline`; best checkpoint by word-anchored validation retrieval."""
    from torch.utils.data import DataLoader
    folder = Path(folder)
    if (folder/'latest.pt').exists(): raise ValueError(f'{folder} already holds a run; choose another --run')
    folder.mkdir(parents=True, exist_ok=True); torch.manual_seed(seed); torch.set_num_threads(threads)
    cache, plan, everyone = setup(cfg, cache_dir, dataset, split, subjects)
    dev_people = set(cache.subjects) if subjects == 'all' else {s for s in cfg.fold_subjects('P1', 'F0', 'validation') if s.startswith(dataset+':')}
    dev_items = plan.items(cache, 'validation', dev_people)
    model = Aligner(np.load(cache.folder/'centroids.npy'), cache.subjects, 'main').to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    loader = DataLoader(PlanStream(cache, plan, set(cache.subjects), seed=seed), batch_size=None, num_workers=workers,
                        persistent_workers=workers > 0, prefetch_factor=4 if workers else None)
    began = time.time(); last = began; best = math.inf; step = 0
    atomic_json(folder/'plan.json', {'dataset': dataset, 'split': split, 'subjects': subjects, 'seconds_per_split': dict(plan.summary),
                                     'train_windows': int(sum(len(v) for v in plan.grid['train'].values())),
                                     'test_candidates': int(sum(len(v) for v in plan.words['test'].values()))})
    status(folder, stage='alignment', dataset=dataset, split=split, subjects=subjects, began=time.strftime('%H:%M'), deadline=time.strftime('%H:%M', time.localtime(deadline)))

    def validate(final=False):
        nonlocal best
        score = identification(model, cache, dev_items, device, limit=limit)['mean_rank_percentile']; minutes = round((time.time()-began)/60, 1)
        append_jsonl(folder/'validation.jsonl', {'step': step, 'minutes': minutes, 'final': final, 'mean_rank_percentile': score})
        saved = {'model': model.state_dict(), 'step': step, 'minutes': minutes, 'subjects': cache.subjects, 'validation': score, 'variant': 'main',
                 'dataset': dataset, 'split': split, 'subject_protocol': subjects}
        torch.save(saved, folder/'latest.pt')
        if score < best or not (folder/'best.pt').exists(): best = min(best, score); torch.save(saved, folder/'best.pt')
        status(folder, validated_step=step, validation=score, best=best if best < math.inf else None)

    for batch in loader:
        now = time.time()
        if now >= deadline or (max_steps and step >= max_steps): break
        for group in optimizer.param_groups: group['lr'] = lr_at((now-began)/max(deadline-began, 1), step)
        model.train(); batch = to(batch, device); wv, en = cache.audio_batch(batch['anchors'])
        E = model.eeg(batch['eeg'], batch['positions'], batch['mask'], batch['subject']); A = model.audio(wv.to(device), en.to(device))
        loss, parts = alignment_losses(model, E, A, batch)
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step(); step += 1
        if step % 25 == 0: append_jsonl(folder/'alignment.jsonl', {'step': step, 'minutes': round((now-began)/60, 1), 'loss': float(loss), **parts})
        if now-last >= validate_minutes*60: last = now; validate()
    validate(final=True); status(folder, stage='alignment done', step=step)


@torch.no_grad()
def evaluate(cfg, cache_dir, folder, out, run, device='cpu', bootstrap=2000):
    """Paper-style test: one candidate per test word; top-1 / top-10 among all test candidates of the dataset."""
    folder, out = Path(folder), Path(out); out.mkdir(parents=True, exist_ok=True)
    saved = torch.load(folder/'best.pt', map_location=device, weights_only=False); dataset, split, subjects = saved['dataset'], saved['split'], saved['subject_protocol']
    cache, plan, everyone = setup(cfg, cache_dir, dataset, split, subjects)
    model = Aligner(np.load(cache.folder/'centroids.npy'), cache.subjects, 'main').to(device); model.load_state_dict(saved['model']); model.eval()
    groups = [('trained-on people', set(cache.subjects))]+([('never-seen people', set(everyone)-set(cache.subjects))] if subjects == 'train' else [])
    summary = {'run': run, 'dataset': dataset, 'split': split, 'subjects': subjects, 'step': saved['step'], 'minutes': saved['minutes'], 'plan': read_json(folder/'plan.json')}
    for name, people in groups:
        result = identification(model, cache, plan.items(cache, 'test', people), device, bootstrap=bootstrap)[dataset]; v = result['own']['training rule']
        summary[name] = {'windows': result['windows'], 'people': result['people'], 'candidates': result['own']['candidates'], 'top1': v['top1'], 'top10': v['top10'],
                         'chance_top1': v['chance_top1'], 'chance_top10': v['chance_top10'], 'mean_rank_percentile': v['mean_rank_percentile'], 'ci95': v.get('ci95')}
    atomic_json(out/'protocol.json', summary)
    paper = PAPER.get(dataset); split_name = {'sentence': '按句子（论文）', 'content': '按内容块（我们）'}[split]; who = {'all': '全部被试参与训练（论文）', 'train': '只用原训练被试（我们）'}[subjects]
    lines = [f"# 协议对照：{run}", '', f"数据集 {dataset}，单独训练；切分 {split_name}；{who}；训练 {saved['step']} 步、{saved['minutes']} 分钟。",
             f"评估按论文取法：每个测试词开始前 0.5 秒起取 3 秒窗口为候选（相邻候选重叠），只有完全对上才算命中。", '',
             '| 人群 | 窗口 | 人 | 候选 | top-1（偶然） | top-10（偶然） | 平均排名百分位（95% 区间） |', '|---|---|---|---|---|---|---|']
    for name, _ in groups:
        v = summary[name]; ci = v['ci95'] or [float('nan')]*2
        lines.append(f"| {name} | {v['windows']} | {v['people']} | {v['candidates']} | {v['top1']:.3f}（{v['chance_top1']:.4f}） | {v['top10']:.3f}（{v['chance_top10']:.4f}） | "
                     f"{v['mean_rank_percentile']:.3f}（{ci[0]:.3f}–{ci[1]:.3f}） |")
    if paper:
        lines += ['', f"论文（{paper['name']}，按句子切分、测试被试也参与训练）：{paper['candidates']} 个候选，top-1 {paper['top1']:.1%}，top-10 {paper['top10']:.1%}。"]
    (out/'protocol.md').write_text('\n'.join(lines)+'\n')
    return summary
