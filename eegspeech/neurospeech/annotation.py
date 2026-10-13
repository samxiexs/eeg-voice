"""Independent transcripts and forced alignment for sources released without them.

Etard: the English stories are chapters of a public-domain book. An ASR draft only
locates each passage and its pauses; the reference words are the book's, never the
recognizer's. Brennan: the official word table. Word and phoneme times come from
Montreal Forced Aligner ``english_us_arpa`` run on pause-bounded chunks, and are
written into the processed manifest together with the tool checksums.
"""
from __future__ import annotations
from collections import Counter
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from eegspeech import ROOT
from eegspeech.utils import digest, read_json
from .storage import dataset_dir, update_manifest

MFA_ENV = ROOT/'.venv-mfa'
MFA_ROOT = ROOT/'artifacts/neurospeech_v2/mfa'
MODEL = 'english_us_arpa'
TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")


def tokens(text):
    """Orthographic word tokens; hyphens and dashes separate words, apostrophes stay inside them."""
    return TOKEN.findall(text.replace('’', "'").replace('‘', "'").replace('_', ' '))


def book_words(path):
    """Body of a Project Gutenberg text, between its START and END markers."""
    text = Path(path).read_text(encoding='utf-8-sig')
    start = re.search(r'\*\*\* ?START OF TH[^\n]*\n', text); stop = re.search(r'\*\*\* ?END OF TH', text)
    if not start or not stop: raise ValueError('Project Gutenberg markers not found')
    return tokens(text[start.end():stop.start()])


def locate(draft, book, gram=5):
    """Book span read in a recording. ``draft`` only votes for the place; it contributes no words.

    Returns (first, stop, pairs): book[first:stop] is the passage and pairs maps
    draft word index -> book word index for words both agree on.
    """
    low = [w.lower() for w in book]; d = [w.lower() for w in draft]
    index = {}
    for i in range(len(low)-gram+1): index.setdefault(tuple(low[i:i+gram]), []).append(i)
    votes = Counter()
    for j in range(len(d)-gram+1):
        for i in index.get(tuple(d[j:j+gram]), ()): votes[(i-j)//50] += 1
    if not votes: raise ValueError('Recording does not occur in the reference text')
    offset = votes.most_common(1)[0][0]*50
    lo = max(0, offset-100); hi = min(len(low), offset+len(d)+200)
    blocks = [b for b in SequenceMatcher(None, d, low[lo:hi], autojunk=False).get_matching_blocks() if b.size]
    pairs = {b.a+k: lo+b.b+k for b in blocks for k in range(b.size)}
    solid = [b for b in blocks if b.size >= 3]
    if not solid: raise ValueError('Recording does not occur in the reference text')
    first, stop = lo+solid[0].b, lo+solid[-1].b+solid[-1].size
    confirmed = len({i for i in pairs.values() if first <= i < stop})
    if confirmed < .85*(stop-first): raise ValueError(f'Recording matches the reference text too loosely: {confirmed}/{stop-first} book words confirmed')
    return first, stop, pairs


def chunks(times, minimum=12., maximum=28., gap=.2):
    """Split word indices at pauses so each aligned piece stays short; times are (start, end) or None.

    A cut is always placed on a word with a known time, so only the first piece can
    begin with unplaced words and every other unplaced word trails its piece.
    """
    known = [i for i, t in enumerate(times) if t is not None]
    if not known: return []
    pieces = []; first = 0; anchor = known[0]
    for a, b in zip(known, known[1:]):
        length = times[a][1]-times[anchor][0]; pause = times[b][0]-times[a][1]
        if (length >= minimum and pause >= gap) or (length >= maximum and pause >= .05):
            pieces.append((first, b)); first = anchor = b
    pieces.append((first, len(times)))
    return pieces


def mfa_models():
    files = {'acoustic': MFA_ROOT/'pretrained_models/acoustic'/f'{MODEL}.zip', 'dictionary': MFA_ROOT/'pretrained_models/dictionary'/f'{MODEL}.dict',
             'g2p': MFA_ROOT/'pretrained_models/g2p'/f'{MODEL}.zip'}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing or not (MFA_ENV/'bin/mfa').exists(): raise FileNotFoundError(f'Montreal Forced Aligner environment or models missing: {missing}')
    return files


def run_mfa(corpus, output):
    files = mfa_models(); env = {**os.environ, 'MFA_ROOT_DIR': str(MFA_ROOT), 'PATH': f'{MFA_ENV}/bin:{os.environ["PATH"]}'}
    env.pop('PYTHONPATH', None); env.pop('VIRTUAL_ENV', None)
    shutil.rmtree(output, ignore_errors=True)
    subprocess.run([str(MFA_ENV/'bin/mfa'), 'align', str(corpus), str(files['dictionary']), str(files['acoustic']), str(output),
                    '--g2p_model_path', str(files['g2p']), '--output_format', 'json', '--clean', '--final_clean', '--beam', '100',
                    '--retry_beam', '400', '-j', '4'], check=True, env=env, cwd=MFA_ROOT)
    version = subprocess.run([str(MFA_ENV/'bin/mfa'), 'version'], capture_output=True, text=True, env=env).stdout.strip()
    return {'tool': 'Montreal Forced Aligner', 'version': version, 'model': MODEL, 'g2p_for_out_of_vocabulary': True,
            'files': {k: {'name': p.name, 'sha256': digest(p)} for k, p in files.items()}}


def merge_labels(labels, words):
    """MFA may split clitics; rebuild one interval per reference word or report failure."""
    result = []; k = 0
    for word in words:
        target = word.lower(); built = ''; begin = None; end = None
        while k < len(labels) and len(built) < len(target):
            a, b, label = labels[k]; built += label.lower(); begin = a if begin is None else begin; end = b; k += 1
        if built != target: return None
        result.append((float(begin), float(end)))
    return result if k == len(labels) else None


def align(dataset, items, root=None):
    """items: {content id: (words, approximate (start, end) or None per word)}. Returns per-content tiers.

    Chunk edges come from the approximate times; every time inside a chunk is MFA's.
    """
    folder = dataset_dir(dataset, root); manifest = read_json(folder/'manifest.json')
    work = ROOT/'artifacts/neurospeech_v2/alignment_work'/dataset; shutil.rmtree(work, ignore_errors=True); corpus = work/'corpus'/'narrator'; corpus.mkdir(parents=True)
    plan = {}
    for content, (words, times) in items.items():
        wave, rate = sf.read(folder/manifest['contents'][content]['path'], dtype='float32')
        if rate != 24000: raise ValueError('Processed audio must be 24 kHz')
        wave16 = resample_poly(wave, 2, 3).astype(np.float32); duration = len(wave)/rate; known = [i for i, t in enumerate(times) if t is not None]
        for n, (a, b) in enumerate(chunks(times)):
            inside = [i for i in known if a <= i < b]
            if not inside: continue
            before = [i for i in known if i < a]; after = [i for i in known if i >= b]
            following = times[after[0]][0] if after else duration
            start = 0. if inside[0] > a else max(0., times[inside[0]][0]-.25, (times[before[-1]][1]+times[inside[0]][0])/2 if before else 0.)
            # Unplaced words at the end of a piece lie before the next placed word.
            stop = max(times[inside[-1]][1], following-.02) if inside[-1] < b-1 else min(duration, times[inside[-1]][1]+.25, (times[inside[-1]][1]+following)/2)
            name = f'{content}__{n:03d}'
            sf.write(corpus/f'{name}.wav', wave16[round(start*16000):round(stop*16000)], 16000, subtype='PCM_16')
            (corpus/f'{name}.lab').write_text(' '.join(w.lower() for w in words[a:b]))
            plan[name] = (content, a, b, start)
    tool = run_mfa(work/'corpus', work/'aligned')
    tiers = {c: {'words': [None]*len(w), 'phonemes': [], 'failed_chunks': []} for c, (w, _) in items.items()}
    for name, (content, a, b, start) in sorted(plan.items()):
        path = work/'aligned'/'narrator'/f'{name}.json'; words = items[content][0][a:b]
        spans = None
        if path.exists():
            grid = json.loads(path.read_text())['tiers']
            spans = merge_labels([e for e in grid['words']['entries'] if e[2] and not e[2].startswith('<')], words)
        if spans is None: tiers[content]['failed_chunks'].append([a, b]); continue
        for i, (x, y) in enumerate(spans): tiers[content]['words'][a+i] = {'start': start+x, 'end': start+y, 'text': words[i]}
        tiers[content]['phonemes'] += [{'start': start+float(x), 'end': start+float(y), 'phone': label}
                                       for x, y, label in grid['phones']['entries'] if label and label not in {'sil', 'sp'}]
    shutil.rmtree(work)
    return tiers, tool


def publish(dataset, content, words, phonemes, annotation, root=None):
    """Replace the tiers of one canonical content; its audio and identity are untouched."""
    folder = dataset_dir(dataset, root); value = read_json(folder/'manifest.json')['contents'][content]
    value = {**value, 'words': words, 'phonemes': phonemes, 'metadata': {**value['metadata'], 'annotation': annotation}}
    value['metadata'].pop('annotation_status', None)
    update_manifest(dataset, ('contents', content), value, root)


def spread(times):
    """Median and 95th percentile of absolute differences in seconds."""
    x = np.abs(np.asarray(times, dtype=float))
    return {'n': int(len(x)), 'median_seconds': float(np.median(x)), 'p95_seconds': float(np.quantile(x, .95))} if len(x) else {'n': 0}


def annotate_brennan(root=None):
    """Phonemes for the official word table; official word times stay the WER reference."""
    manifest = read_json(dataset_dir('brennan2019', root)/'manifest.json')
    items = {c: ([w['text'] for w in v['words']], [(w['start'], w['end']) for w in v['words']]) for c, v in manifest['contents'].items()}
    tiers, tool = align('brennan2019', items, root); report = {}
    for content, tier in tiers.items():
        official = manifest['contents'][content]['words']
        errors = [m['start']-o['start'] for m, o in zip(tier['words'], official) if m is not None]
        quality = {'method': 'MFA word onsets against the official word table of the same audio', 'word_onset_difference': spread(errors),
                   'aligned_words': sum(m is not None for m in tier['words']), 'words': len(official), 'failed_chunks': tier['failed_chunks']}
        publish('brennan2019', content, official, tier['phonemes'], {**tool, 'tiers': 'phonemes from MFA; words and word times from the official table',
                'quality': quality, 'manual_boundary_review': False}, root)
        report[content] = quality
    return report


def annotate_etard(drafts, book, root=None, source=None):
    """Book-text transcripts with MFA word and phoneme times for the sixteen English story parts.

    drafts: {content id: [{'text': word, 'timestamp': [start, end]}, ...]} from any recognizer.
    """
    reference = book_words(book); manifest = read_json(dataset_dir('etard', root)/'manifest.json'); items = {}; located = {}
    for content in manifest['contents']:
        draft = [(t, c['timestamp']) for c in drafts[content] for t in tokens(c['text'])]
        first, stop, pairs = locate([t for t, _ in draft], reference)
        words = reference[first:stop]; times = [None]*len(words)
        for j, i in pairs.items():
            stamp = draft[j][1]
            if first <= i < stop and stamp[0] is not None and stamp[1] is not None and stamp[1] > stamp[0]: times[i-first] = (float(stamp[0]), float(stamp[1]))
        items[content] = (words, times)
        located[content] = {'book_word_range': [first, stop], 'draft_words': len(draft), 'draft_words_agreeing_with_book': len(pairs),
                            'book_words_confirmed_by_draft': sum(t is not None for t in times), 'book_words': len(words)}
    tiers, tool = align('etard', items, root); report = {}
    for content, tier in tiers.items():
        approximate = items[content][1]
        errors = [m['start']-a[0] for m, a in zip(tier['words'], approximate) if m is not None and a is not None]
        words = [m for m in tier['words'] if m is not None]
        quality = {**located[content], 'method': 'MFA word onsets against recognizer word times (consistency only, no independent timing exists)',
                   'word_onset_difference': spread(errors), 'aligned_words': len(words), 'failed_chunks': tier['failed_chunks']}
        publish('etard', content, words, tier['phonemes'], {**tool, 'transcript_source': source or str(book),
                'transcript_rule': 'words are the published book text; the recognizer draft only located the passage and its pauses',
                'tiers': 'words, word times and phonemes from MFA', 'quality': quality, 'manual_boundary_review': False}, root)
        report[content] = quality
    return report


BOOK = {'title': 'The Children of Odin: The Book of Northern Myths (Padraic Colum), Project Gutenberg eBook 24737',
        'url': 'https://www.gutenberg.org/cache/epub/24737/pg24737.txt',
        'sha256': '530b56ba38d9c1fdc31652ec73665afeacfda7e6d9514cf8977171873a04508c',
        'evidence': 'original Zenodo 7086209 stimulus files are named odin_<chapter>_<part>; recognizer drafts match the book text'}


def etard_reference(root=None, device='cpu'):
    """Fetch the book once and draft every story part; both are kept beside the data as provenance."""
    from .storage import download_file
    folder = dataset_dir('etard', root); store = folder/'metadata/reference_text'; (store/'drafts').mkdir(parents=True, exist_ok=True)
    book = store/'children_of_odin_pg24737.txt'
    if not book.exists(): download_file(BOOK['url'], book, sha256=BOOK['sha256'])
    drafts = {}; asr = None
    for content, value in sorted(read_json(folder/'manifest.json')['contents'].items()):
        path = store/'drafts'/f'{content}.json'
        if not path.exists():
            if asr is None:
                from transformers import pipeline
                asr = pipeline('automatic-speech-recognition', model=str(ROOT/'models/whisper-small.en'), device=device)
            wave, rate = sf.read(folder/value['path'], dtype='float32'); wave = resample_poly(wave, 2, 3).astype(np.float32); words = []
            for first in range(0, len(wave), 28*16000):   # native 30 s recognizer window, no overlap to merge
                piece = wave[first:first+28*16000]
                if len(piece) < 8000: continue
                for q in asr({'raw': piece, 'sampling_rate': 16000}, return_timestamps='word')['chunks']:
                    a, b = q['timestamp']
                    words.append({'text': q['text'], 'timestamp': [None if a is None else a+first/16000, None if b is None else b+first/16000]})
            path.write_text(json.dumps({'id': content, 'recognizer': 'openai/whisper-small.en (locator only)', 'chunks': words}))
        drafts[content] = json.loads(path.read_text())['chunks']
    return drafts, book


def untranscribed_sound(dataset, content, root=None, level_db=-12., minimum=.15, pad=.1, join=.3):
    """Speech-level sound inside pauses of the released word tier: audio its transcript does not account for.

    Level is relative to the median RMS of the transcribed words. This finds
    backchannels, laughter and cross-talk between words; speech that overlaps a
    transcribed word cannot be seen this way.
    """
    folder = dataset_dir(dataset, root); value = read_json(folder/'manifest.json')['contents'][content]; words = value['words']
    wave, rate = sf.read(folder/value['path'], dtype='float32')
    rms = lambda a, b: float(np.sqrt(np.mean(np.square(wave[int(a*rate):max(int(b*rate), int(a*rate)+1)]))+1e-12))
    speech = float(np.median([rms(w['start'], w['end']) for w in words])); found = []
    for a, b in zip(words, words[1:]):
        if b['start']-a['end'] >= minimum and 20*np.log10(rms(a['end'], b['start'])/speech) >= level_db:
            lo, hi = max(0., a['end']-pad), min(value['duration'], b['start']+pad)
            if found and lo-found[-1][1] <= join: found[-1][1] = round(hi, 3)
            else: found.append([round(lo, 3), round(hi, 3)])
    return found, {'method': 'speech-level sound in pauses of the released word tier', 'level_db_re_median_word_rms': level_db,
                   'minimum_pause_seconds': minimum, 'pad_seconds': pad, 'audio_sha256': value['sha256'],
                   'limitation': 'simultaneous speech during a transcribed word is not detected; the source releases no speaker turns'}


def lock_content_mask(dataset, content, kind, config=None, root=None):
    """Write the automatically derived intervals into the reviewed-mask file used by the registry."""
    path = Path(config or ROOT/'configs/neurospeech_v2/content_masks.json'); masks = read_json(path) if path.exists() else {}
    intervals, method = untranscribed_sound(dataset, content, root)
    masks[f'{dataset}:{content}'] = {'kind': kind, 'status': 'locked', 'human_review': False, **method, 'intervals': intervals,
                                     'note': 'Derived automatically from the released annotation and audio; edit the intervals and keep status locked after any manual review.'}
    path.write_text(json.dumps(masks, indent=2)+'\n')
    return masks[f'{dataset}:{content}']
