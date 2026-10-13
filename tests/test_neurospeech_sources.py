"""Source-reader contracts on tiny synthetic fixtures; no download, no real recording."""
import numpy as np
import pytest
from scipy import signal
from scipy.io import savemat
from eegspeech.neurospeech.sources import kymata
from eegspeech.neurospeech.sources.karaone import phase_indices


def speech_like(seconds,seed=3):
    """Aperiodic syllable-rate envelope on a voiced carrier; periodic envelopes would alias the lag."""
    rng=np.random.default_rng(seed);t=np.arange(seconds*1000)/1000
    syllables=np.clip(signal.fftconvolve(rng.normal(size=len(t)),signal.firwin(801,6,fs=1000),mode='same'),0,None)
    return syllables*np.sin(2*np.pi*140*t)+.001*rng.normal(size=len(t))


def played(audio,offset,drift,seed=5):
    """Monitor as recorded: PRE samples of lead-in, then audio on a slower clock."""
    rate=1000+drift;n=int(len(audio)*rate/1000)+3000
    position=(np.arange(n)-kymata.PRE-offset)*1000/rate
    wave=np.interp(position,np.arange(len(audio)),audio,left=0,right=0)
    return wave+.02*np.random.default_rng(seed).normal(size=n)


def test_clock_fit_recovers_onset_and_drift_from_monitor():
    audio=speech_like(130)
    fit=kymata.clock_fit(kymata.envelope(played(audio,19,.54)),kymata.envelope(audio))
    assert fit['status']=='measured-aligned' and fit['accepted_chunks']>=10
    assert abs(fit['offset_ms']-19)<1.5 and abs(fit['drift_ms_per_second']-.54)<.03
    assert fit['maximum_residual_seconds']<.003 and fit['audio_monitor_excluded']


def test_playback_slip_is_confined_to_its_content_block():
    audio=speech_like(190);monitor=played(audio,19,.54);slip=kymata.PRE+19+int(100*1000.54)
    monitor=np.concatenate((monitor[:slip],np.zeros(14),monitor[slip:]))
    fit=kymata.clock_fit(kymata.envelope(monitor),kymata.envelope(audio))
    assert fit['status']=='measured-piecewise'
    before,during,after=(kymata.block_lag(fit,start,60) for start in (0,60,120))
    assert during is None and abs(before-19)<1.5 and abs(after-(19+.54*120+14))<1.5


def test_clock_fit_rejects_monitor_without_the_stimulus():
    audio=speech_like(130)
    with pytest.raises(ValueError,match='does not follow'):
        kymata.clock_fit(kymata.envelope(played(speech_like(130,seed=99),19,.54)),kymata.envelope(audio))


def test_released_tiers_become_bounded_tokens(tmp_path):
    (tmp_path/'w.txt').write_text('<sp>\nice\ncream\n<sp>\nspoons\n');(tmp_path/'t.txt').write_text('0.0\n0.5\n0.9\n1.4\n2.0\n')
    words=kymata.tokens(tmp_path/'w.txt',tmp_path/'t.txt',3.,'text')
    assert words==[{'start':.5,'end':.9,'text':'ice'},{'start':.9,'end':1.4,'text':'cream'},{'start':2.,'end':3.,'text':'spoons'}]
    with pytest.raises(ValueError):kymata.tokens(tmp_path/'w.txt',tmp_path/'t.txt',1.5,'text')


def test_second_session_is_the_same_person():
    assert kymata.person('sub-E3b')=='sub-E3' and kymata.person('sub-E13')=='sub-E13'


def trial(start):
    return {'rest':[start+1,start+5000],'prompt':[start+5000,start+7000],'imagined':[start+7001,start+12000],'spoken':[start+12100,start+14000]}


def write_epochs(path,trials,speaking):
    cells=lambda rows:np.array([np.array(r) for r in rows]+[None],dtype=object)[:-1]
    savemat(path,{'clearing_inds':cells([t['rest'] for t in trials]),'thinking_inds':cells([t['imagined'] for t in trials]),'speaking_inds':cells(speaking)})


def test_karaone_places_speaking_ranges_by_position_and_withholds_empty_slots(tmp_path):
    trials=[trial(i*15000) for i in range(4)]
    # Source order is unreliable: trial 1 and 3 swap places, trial 2 repeats its anchors instead.
    order=[trials[0]['prompt'],trials[0]['spoken'],trials[3]['prompt'],trials[3]['spoken'],
           trials[2]['rest'],[trials[2]['imagined'][0],trials[2]['imagined'][1]+300],trials[1]['prompt'],trials[1]['spoken']]
    write_epochs(tmp_path/'epoch_inds.mat',trials,order)
    states,withheld,unplaced=phase_indices(tmp_path/'epoch_inds.mat',4)
    assert set(states[2])=={'rest','imagined'} and set(withheld[2])=={'stimulus_preparation','articulated'} and len(unplaced)==2
    for i in (0,1,3):
        assert not withheld[i] and states[i]['stimulus_preparation'][1]==trials[i]['prompt'][1] and states[i]['articulated']==[trials[i]['spoken'][0]-1,trials[i]['spoken'][1]]
    # MATLAB inclusive bounds share one sample; it stays with the earlier state.
    assert states[0]['rest']==[0,5000] and states[0]['stimulus_preparation']==[5000,7000]
    for kept in states:
        ranges=list(kept.values());assert all(a[1]<=b[0] for a,b in zip(ranges,ranges[1:]))


def test_reference_text_supplies_the_words_and_the_draft_only_locates_them():
    from eegspeech.neurospeech import annotation as A
    book=A.tokens(' '.join(f'filler{i} word' for i in range(200))+" Always there had been war between the Giants and the Gods—between the Giants who would have destroyed the world, O Odin's children "+' '.join(f'tail{i} end' for i in range(200)))
    draft=A.tokens('always there had been war between the giants and the gods between the giants who would have destroyed the world oh')
    first,stop,pairs=A.locate(draft,book)
    assert book[first:stop][:3]==['Always','there','had'] and book[stop-1]=='world' and 'oh' not in [w.lower() for w in book[first:stop]]
    assert all(book[i].lower()==draft[j] for j,i in pairs.items())
    with pytest.raises(ValueError):A.locate(A.tokens('completely different words that never occur anywhere in this text at all'),book)


def test_alignment_pieces_cut_at_pauses_and_clitics_are_rejoined():
    from eegspeech.neurospeech import annotation as A
    times=[(i*.4,i*.4+.3) for i in range(40)]+[None,None]+[(17.5+i*.4,17.8+i*.4) for i in range(40)]
    pieces=A.chunks(times)
    assert pieces[0]==(0,42) and pieces[-1][1]==len(times) and all(a<b for a,b in pieces)
    assert A.merge_labels([[0,1,'alice'],[1,1.2,"'s"],[2,3,'cat']],["Alice's",'cat'])==[(0.,1.2),(2.,3.)]
    assert A.merge_labels([[0,1,'alice'],[2,3,'dog']],["Alice's",'cat']) is None


def test_stimulus_onsets_are_found_on_the_monitor_when_the_trigger_is_missing():
    audio=speech_like(130);gap=np.zeros(400000)
    monitor=np.concatenate((np.zeros(5000),audio,gap,audio,np.zeros(3000)))+.02*np.random.default_rng(1).normal(size=5000+2*len(audio)+len(gap)+3000)
    class Raw:
        def get_data(self,picks):return monitor[None]
    found=kymata.monitor_onsets(Raw(),['MISC006'],kymata.envelope(audio))
    assert len(found)==2 and abs(found[0]-5000)<=5 and abs(found[1]-(5000+len(audio)+len(gap)))<=5
