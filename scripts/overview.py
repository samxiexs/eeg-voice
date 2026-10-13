"""outputs/index.html: the latest reconstructions next to the earlier versions.

    python scripts/overview.py

Items (outputs/reconstruct/ vs outputs/reconstruct_v3_decision/, where the encoder's decision was rendered)
and sentences (outputs/sentences/, outputs/sentences_deep/: speech generated from the EEG embedding; the
earlier versions v1 and v2 only ranked candidate sentences and played the top one).
"""
from __future__ import annotations

import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eegspeech import ROOT                                                     # noqa: E402

OUT = ROOT / 'outputs'
KINDS = ('eeg', 'wrong', 'prior', 'oracle')
PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Reconstruction</title><style>
body{{font:14px/1.45 system-ui,sans-serif;margin:16px;max-width:1500px}} table{{border-collapse:collapse;margin:8px 0}}
td,th{{border:1px solid #ccc;padding:4px 6px;vertical-align:top;text-align:left}} .num{{text-align:right}}
.muted{{color:#666}} b{{color:#000}}</style></head><body>{body}</body></html>"""


def load(path):
    return json.load(open(path)) if path.exists() else {}


def mean(v, digits=3):
    return f'{v["mean"]:.{digits}f}' if isinstance(v, dict) and 'mean' in v else '-'


def items():
    rows = []
    for label, folder, how in (('latest', 'reconstruct', 'posterior mean of the speech embedding (no item chosen)'),
                               ('2026-10-06', 'reconstruct_v3_decision', 'the encoder\'s decision (an item chosen)')):
        for name, d in load(OUT / folder / 'summary.json').items():
            r = d['results']
            speech = lambda kind: ' / '.join(mean(r[kind].get(m, {})) for m in ('listener', 'mcd', 'hubert'))
            vs = r['eeg'].get('listener', {}).get('vs_wrong', {})
            rows.append(f'<tr><td>{label}</td><td>{html.escape(how)}</td><td>{name} ({d["chance"]:.2f})</td>'
                        f'<td class="num">{mean(r["encoder"])}</td><td class="num"><b>{speech("eeg")}</b></td>'
                        f'<td class="num">{speech("wrong")}</td><td class="num">{speech("prior")}</td>'
                        f'<td class="num">{speech("oracle")}</td><td class="num">{vs.get("people_better", "-")}/'
                        f'{vs.get("people", "-")}, p {vs.get("p_wilcoxon", float("nan")):.3g}</td>'
                        f'<td>folds {d["folds"]}</td></tr>')
    return ('<h3>Items: BCI2020, Thinking Out Loud (<a href="reconstruct/index.html">pages</a>)</h3>'
            '<table><tr><th>run</th><th>speech generated from</th><th>dataset (chance)</th><th>encoder accuracy</th>'
            '<th>from EEG: identified by Whisper / MCD / HuBERT</th><th>wrong trial</th><th>prior</th>'
            '<th>oracle (ceiling)</th><th>EEG > wrong trial (people, p)</th><th></th></tr>' + ''.join(rows) + '</table>')


def sentences():
    rows = []
    for label, folder in (('latest, linear encoder', 'sentences'), ('latest, deep encoder', 'sentences_deep')):
        for name, d in load(OUT / folder / 'summary.json').items():
            r, e = d['results']['reconstruction'], d['results']['encoder']
            vs = r['eeg_vs_wrong']
            rows.append(f'<tr><td>{label}</td><td>generated from the EEG embedding</td>'
                        f'<td class="num">{mean(e["percentile"], 4)}</td>'
                        + ''.join(f'<td class="num">{"<b>" if k == "eeg" else ""}{mean(r[k]["percentile"], 4)}'
                                  f'{"</b>" if k == "eeg" else ""}</td>' for k in KINDS)
                        + f'<td class="num">{mean(r["eeg"]["top10"])} ({r["eeg"]["top10"].get("chance", float("nan")):.3f})</td>'
                        f'<td class="num">{mean(r["eeg"]["duration_r"])} / {mean(r["oracle"]["duration_r"])}</td>'
                        f'<td class="num">{r["eeg"]["heard_as_true"]:.3f} / {r["oracle"]["heard_as_true"]:.3f}</td>'
                        f'<td class="num">{vs["people_better"]}/{vs["people"]}, p {vs.get("p_wilcoxon", float("nan")):.3g}</td>'
                        f'<td>folds {d["folds"]}, {d["rendered"]} trials rendered</td></tr>')
    for label, folder in (('2026-10-06 v2, time course', 'sentences_v2_time'), ('2026-10-06 v1, whole trial', 'sentences_v1_whole_trial')):
        for name, d in load(OUT / folder / 'summary.json').items():
            r = d['results']
            rows.append(f'<tr class="muted"><td>{label}</td><td>none: the top candidate\'s synthetic speech was played</td>'
                        f'<td class="num">{mean(r["percentile"], 4)}</td>' + '<td>-</td>' * 4
                        + f'<td class="num">-</td><td>-</td><td>-</td><td>-</td><td>folds {d["folds"]}</td></tr>')
    return ('<h3>Sentences: Chisco, ~130 candidate sentences per run, none seen in training '
            '(<a href="sentences/index.html">linear</a>, <a href="sentences_deep/index.html">deep</a>)</h3>'
            '<table><tr><th>run</th><th>speech</th><th>encoder ranking (diagnostic)</th>'
            + ''.join(f'<th>generated speech, rank percentile of the true sentence: {k}</th>' for k in KINDS)
            + '<th>top-10 from EEG (chance)</th><th>duration r, EEG / oracle</th><th>Whisper heard the true sentence, '
              'EEG / oracle</th><th>EEG > wrong trial (people, p)</th><th></th></tr>' + ''.join(rows) + '</table>'
            '<p class="muted">Rank percentile: where the generated speech, embedded by HuBERT, puts the true sentence '
            'among the run\'s sentences (1 best, 0.5 chance). Oracle: the renderer\'s speech from the true sentence\'s '
            'point of the space, for a sentence it never heard: the ceiling of reconstruction through this space.</p>')


def main():
    body = ('<h2>Imagined speech reconstructed from EEG (within person, held-out blocks)</h2>'
            '<p>Latest runs first. "Generated from" says what the speech generator is given: in the latest runs a '
            'continuous point of the speech space computed from the EEG, so no item or sentence is chosen anywhere '
            'on the way to the speech; earlier runs rendered a chosen item or played a chosen sentence.</p>'
            + items() + sentences())
    (OUT / 'index.html').write_text(PAGE.format(body=body))
    print('->', OUT / 'index.html')


if __name__ == '__main__':
    main()
