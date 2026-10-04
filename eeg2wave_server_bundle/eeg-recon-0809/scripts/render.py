"""Speech for decoded items: EEG -> item -> waveform.

    python scripts/render.py items  thinking_out_loud
    python scripts/render.py decode thinking_out_loud --run outputs/imagery/pretrained_f0 --subject sub-03 [--shots 5]

``items`` synthesises one waveform per vocabulary item with the macOS ``say`` voices of
``configs/plan.yaml`` (``artifacts/render/<dataset>/``).  ``decode`` runs a trained model on
one held-out person's imagined trials, picks an item per trial (dataset head, or nearest
prototype of ``--shots`` labelled trials of that person, which are then left out) and writes
the sequence of decoded items as speech plus a CSV of true / decoded items and confidences.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT                                                    # noqa: E402
from eegspeech.data import TrialSource                                         # noqa: E402
from eegspeech.evaluation import embed_trials, person_vector                    # noqa: E402
from eegspeech.model import Model                                              # noqa: E402
from eegspeech.store import open_store                                         # noqa: E402

RENDER = ROOT / 'artifacts' / 'render'
RATE = 16000


def item_path(dataset, item):
    return RENDER / dataset / (item.strip('/').replace(' ', '_') + '.wav')


def items(dataset, config):
    spec = config['render'].get(dataset)
    if spec is None:
        raise SystemExit(f'no render entry for {dataset} in configs/plan.yaml')
    for item, text in spec['text'].items():
        path = item_path(dataset, item)
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['say', '-v', spec['voice'], '-o', str(path), '--file-format=WAVE', f'--data-format=LEI16@{RATE}',
                        str(text)], check=True)
        print(path)


def decode(args, config):
    model, payload = Model.load(Path(args.run) / 'model.pt')
    device = torch.device('cpu')
    spec = config['imagery']['sources'][args.dataset]
    store = open_store(args.dataset)
    if args.subject not in payload['splits'].get(args.dataset, {}).get('test', []):
        print(f'warning: {args.subject} was not held out by this run')
    source = TrialSource(store, [args.subject], window_s=spec['window'], modalities=spec['modalities'], align=payload['cfg']['align'])
    rows = source.rows
    target = rows.index[rows.modality_name == spec['target']].to_numpy()
    person = person_vector(model, source, args.subject, device)
    logits, embeddings = embed_trials(model, source, target, device, head=args.dataset if args.dataset in model.heads else None,
                                      person=person)
    labels = rows.item.to_numpy()[target]
    keep = np.ones(len(target), bool)
    if args.shots:
        rng = np.random.default_rng(0)
        classes = np.unique(labels)
        support = np.concatenate([rng.choice(np.flatnonzero(labels == k), args.shots, replace=False) for k in classes])
        keep[support] = False
        prototypes = torch.stack([torch.nn.functional.normalize(embeddings[support[labels[support] == k]].mean(0), dim=-1)
                                  for k in classes])
        scores = torch.softmax(embeddings @ prototypes.T / .1, -1)
        predicted, confidence = classes[scores.argmax(1).numpy()], scores.max(1).values.numpy()
    else:
        scores = logits.exp()
        predicted, confidence = scores.argmax(1).numpy(), scores.max(1).values.numpy()
    out = Path(args.run) / 'render'
    out.mkdir(exist_ok=True)
    gap = np.zeros(int(.4 * RATE))
    audio, table = [], []
    for i in np.flatnonzero(keep):
        name = store.items[int(predicted[i])]
        wave, rate = sf.read(item_path(args.dataset, name))
        assert rate == RATE
        audio += [wave, gap]
        table.append(dict(trial=int(target[i]), true=store.items[int(labels[i])], decoded=name,
                          confidence=round(float(confidence[i]), 3)))
    stem = f'{args.dataset}_{args.subject}' + (f'_{args.shots}shot' if args.shots else '')
    sf.write(out / f'{stem}.wav', np.concatenate(audio), RATE)
    with open(out / f'{stem}.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader(); writer.writerows(table)
    accuracy = np.mean([r['true'] == r['decoded'] for r in table])
    print(f'{len(table)} trials, accuracy {accuracy:.3f} (chance {1 / len(store.items):.3f}) -> {out / stem}.wav')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['items', 'decode'])
    parser.add_argument('dataset')
    parser.add_argument('--run')
    parser.add_argument('--subject')
    parser.add_argument('--shots', type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load(open(ROOT / 'configs' / 'plan.yaml'))
    if args.command == 'items':
        items(args.dataset, config)
    else:
        decode(args, config)
