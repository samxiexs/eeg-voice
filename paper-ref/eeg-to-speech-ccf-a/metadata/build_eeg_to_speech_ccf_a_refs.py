from __future__ import annotations

import csv
import shutil
import textwrap
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "pdf"
METADATA_DIR = ROOT / "metadata"
REPO_ROOT = ROOT.parents[1]


PAPERS = [
    {
        "key": "gao_neurosonic_2026",
        "title": "NeuroSonic: Conditional Flow Matching for EEG-to-Speech Reconstruction",
        "authors": "Wenhao Gao; Yifan Wang; Yijia Ma; Carl Yang; Wen Li; Chenyu You",
        "year": "2026",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P0 direct EEG-to-speech",
        "filename": "2026_arXiv_NeuroSonic_Conditional_Flow_Matching_EEG_to_Speech.pdf",
        "url": "https://arxiv.org/abs/2606.24087",
        "pdf_urls": ["https://arxiv.org/pdf/2606.24087.pdf"],
        "relevance": "Most direct current reference for EEG-conditioned speech reconstruction using conditional flow matching.",
    },
    {
        "key": "lee_neurotalk_2023",
        "title": "Towards Voice Reconstruction from EEG during Imagined Speech",
        "authors": "Young-Eun Lee; Seo-Hyun Lee; Sang-Ho Kim; Seong-Whan Lee",
        "year": "2023",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P0 direct EEG-to-speech",
        "filename": "2023_arXiv_NeuroTalk_Voice_Reconstruction_from_EEG.pdf",
        "url": "https://arxiv.org/abs/2301.07173",
        "pdf_urls": ["https://arxiv.org/pdf/2301.07173.pdf"],
        "local_source": REPO_ROOT / "paper-ref/unclassified-root-papers/Lee 等 - 2023 - Towards Voice Reconstruction from EEG during Imagined Speech.pdf",
        "relevance": "NeuroTalk baseline for imagined-speech EEG-to-voice reconstruction and domain adaptation from spoken EEG.",
    },
    {
        "key": "defossez_decoding_2022",
        "title": "Decoding speech perception from non-invasive brain recordings",
        "authors": "Alexandre Defossez; Charlotte Caucheteux; Jeremy Rapin; Ori Kabeli; Jean-Remi King",
        "year": "2022",
        "venue": "arXiv / Nature Machine Intelligence-adjacent line",
        "ccf_a": "No",
        "group": "P0 direct EEG-to-speech",
        "filename": "2022_arXiv_Decoding_Speech_Perception_Non_Invasive_Brain_Recordings.pdf",
        "url": "https://arxiv.org/abs/2208.12266",
        "pdf_urls": ["https://arxiv.org/pdf/2208.12266.pdf"],
        "local_source": REPO_ROOT / "paper-ref/unclassified-root-papers/Défossez 等 - 2022 - Decoding speech perception from non-invasive brain recordings.pdf",
        "relevance": "Core non-invasive M/EEG speech decoding reference; supports contrastive speech-representation retrieval baselines.",
    },
    {
        "key": "park_eeg_to_voice_2025",
        "title": "EEG-to-Voice Decoding of Spoken and Imagined speech Using Non-Invasive EEG",
        "authors": "Hanbeot Park; Yunjeong Cho; Hunhee Kim",
        "year": "2025",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P0 direct EEG-to-speech",
        "filename": "2025_arXiv_EEG_to_Voice_Spoken_Imagined_Speech.pdf",
        "url": "https://arxiv.org/abs/2512.22146",
        "pdf_urls": ["https://arxiv.org/pdf/2512.22146.pdf"],
        "relevance": "Direct open-loop EEG-to-mel/vocoder reconstruction for spoken and imagined speech; closest to the current FEIS/KaraOne evaluation style.",
    },
    {
        "key": "baevski_wav2vec2_2020",
        "title": "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations",
        "authors": "Alexei Baevski; Henry Zhou; Abdelrahman Mohamed; Michael Auli",
        "year": "2020",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P0 audio codec/token target",
        "filename": "2020_NeurIPS_wav2vec_2_0_Speech_Representations.pdf",
        "url": "https://arxiv.org/abs/2006.11477",
        "pdf_urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf",
            "https://arxiv.org/pdf/2006.11477.pdf",
        ],
        "relevance": "Canonical self-supervised speech representation target for EEG-to-speech semantic alignment.",
    },
    {
        "key": "radford_whisper_2023",
        "title": "Robust Speech Recognition via Large-Scale Weak Supervision",
        "authors": "Alec Radford; Jong Wook Kim; Tao Xu; Greg Brockman; Christine McLeavey; Ilya Sutskever",
        "year": "2023",
        "venue": "ICML",
        "ccf_a": "Yes",
        "group": "P0 audio codec/token target",
        "filename": "2023_ICML_Whisper_Robust_Speech_Recognition.pdf",
        "url": "https://proceedings.mlr.press/v202/radford23a.html",
        "pdf_urls": ["https://proceedings.mlr.press/v202/radford23a/radford23a.pdf"],
        "relevance": "Robust speech encoder and ASR sanity-check reference for reconstructed wav intelligibility.",
    },
    {
        "key": "chen_beats_2023",
        "title": "BEATs: Audio Pre-Training with Acoustic Tokenizers",
        "authors": "Sanyuan Chen; Yu Wu; Chengyi Wang; Shujie Liu; Daniel Tompkins; Zhuo Chen; Furu Wei",
        "year": "2023",
        "venue": "ICML / arXiv",
        "ccf_a": "Yes",
        "group": "P0 audio codec/token target",
        "filename": "2023_ICML_BEATs_Audio_Pretraining_Acoustic_Tokenizers.pdf",
        "url": "https://arxiv.org/abs/2212.09058",
        "pdf_urls": ["https://arxiv.org/pdf/2212.09058.pdf"],
        "relevance": "Acoustic tokenizer reference for non-speech and speech auditory token targets.",
    },
    {
        "key": "kumar_rvqgan_2023",
        "title": "High-Fidelity Audio Compression with Improved RVQGAN",
        "authors": "Rithesh Kumar; Prem Seetharaman; Alejandro Luebs; Ishaan Kumar; Kundan Kumar",
        "year": "2023",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P0 audio codec/token target",
        "filename": "2023_NeurIPS_Improved_RVQGAN_Audio_Compression.pdf",
        "url": "https://arxiv.org/abs/2306.06546",
        "pdf_urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/file/58d0e78cf042af5876e12661087bea12-Paper-Conference.pdf",
            "https://arxiv.org/pdf/2306.06546.pdf",
        ],
        "relevance": "High-fidelity neural audio codec baseline for codec-token waveform rendering.",
    },
    {
        "key": "ju_naturalspeech3_2024",
        "title": "NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models",
        "authors": "Zeqian Ju; Yuancheng Wang; Kai Shen; Xu Tan; Detai Xin; Dongchao Yang; Yanqing Liu; Yichong Leng; Kaitao Song; Siliang Tang; Zhizheng Wu; Tao Qin; Xiang-Yang Li; Wei Ye; Shikun Zhang; Jiang Bian; Lei He; Jinyu Li; Sheng Zhao",
        "year": "2024",
        "venue": "ICML",
        "ccf_a": "Yes",
        "group": "P0 audio codec/token target",
        "filename": "2024_ICML_NaturalSpeech_3_Factorized_Codec_Diffusion.pdf",
        "url": "https://proceedings.mlr.press/v235/ju24b.html",
        "pdf_urls": [
            "https://proceedings.mlr.press/v235/ju24b/ju24b.pdf",
            "https://arxiv.org/pdf/2403.03100.pdf",
        ],
        "relevance": "Best factorized content/prosody/timbre/acoustic codec reference for EEG token decomposition.",
    },
    {
        "key": "ye_xcodec_2025",
        "title": "Codec Does Matter: Exploring the Semantic Shortcoming of Codec for Audio Language Model",
        "authors": "Zhen Ye; Peiwen Sun; Jiahe Lei; Hongzhan Lin; Xu Tan; Zhe Dai; Qiuqiang Kong; Jianyi Chen; Jiahao Pan; Qifeng Liu; Yike Guo; Wei Xue",
        "year": "2025",
        "venue": "AAAI",
        "ccf_a": "Yes",
        "group": "P0 audio codec/token target",
        "filename": "2025_AAAI_X_Codec_Semantic_Audio_Codec.pdf",
        "url": "https://arxiv.org/abs/2408.17175",
        "pdf_urls": ["https://arxiv.org/pdf/2408.17175.pdf"],
        "relevance": "Semantic-enhanced codec reference; useful for bridging EEG semantic targets and acoustic tokens.",
    },
    {
        "key": "le_voicebox_2023",
        "title": "Voicebox: Text-Guided Multilingual Universal Speech Generation at Scale",
        "authors": "Matthew Le; Apoorv Vyas; Bowen Shi; Brian Karrer; Leda Sari; Rashel Moritz; Mary Williamson; Vimal Manohar; Yossi Adi; Jay Mahadeokar; Wei-Ning Hsu",
        "year": "2023",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2023_NeurIPS_Voicebox_Universal_Speech_Generation.pdf",
        "url": "https://arxiv.org/abs/2306.15687",
        "pdf_urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/file/2d8911db9ecedf866015091b28946e15-Paper-Conference.pdf",
            "https://arxiv.org/pdf/2306.15687.pdf",
        ],
        "relevance": "Flow-matching speech generation reference for using partial EEG-derived conditions to realize speech.",
    },
    {
        "key": "li_styletts2_2023",
        "title": "StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models",
        "authors": "Yinghao Aaron Li; Cong Han; Vinay S. Raghavan; Gavin Mischler; Nima Mesgarani",
        "year": "2023",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2023_NeurIPS_StyleTTS_2_Style_Diffusion.pdf",
        "url": "https://arxiv.org/abs/2306.07691",
        "pdf_urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/file/3eaad2a0b62b5ed7a2e66c2188bb1449-Paper-Conference.pdf",
            "https://arxiv.org/pdf/2306.07691.pdf",
        ],
        "relevance": "Style/prosody diffusion decoder reference for voice realization from high-level conditions.",
    },
    {
        "key": "fang_daspeech_2023",
        "title": "DASpeech: Directed Acyclic Transformer for Fast and High-quality Speech-to-Speech Translation",
        "authors": "Qingkai Fang; Yan Zhou; Yang Feng",
        "year": "2023",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2023_NeurIPS_DASpeech_Two_Stage_Speech_Decoder.pdf",
        "url": "https://arxiv.org/abs/2310.07403",
        "pdf_urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2023/file/e5b1c0d4866f72393c522c8a00eed4eb-Paper-Conference.pdf",
            "https://arxiv.org/pdf/2310.07403.pdf",
        ],
        "relevance": "Two-stage linguistic-to-acoustic decoding design, useful when EEG provides incomplete high-level speech structure.",
    },
    {
        "key": "kim_pflow_2023",
        "title": "P-Flow: A Fast and Data-Efficient Zero-Shot TTS through Speech Prompting",
        "authors": "Sungwon Kim; Kevin Shih; Rohan Badlani; Joao Felipe Santos; Evelina Bakhturina; Mikyas Desta; Rafael Valle; Sungroh Yoon; Bryan Catanzaro",
        "year": "2023",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2023_NeurIPS_P_Flow_Zero_Shot_TTS.pdf",
        "url": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Abstract-Conference.html",
        "pdf_urls": ["https://proceedings.neurips.cc/paper_files/paper/2023/file/eb0965da1d2cb3fbbbb8dbbad5fa0bfc-Paper-Conference.pdf"],
        "relevance": "Prompt-conditioned flow decoder reference for fast voice-conditioned waveform realization.",
    },
    {
        "key": "ye_comospeech_2023",
        "title": "CoMoSpeech: One-Step Speech and Singing Voice Synthesis via Consistency Model",
        "authors": "Zhen Ye; Wei Xue; Xu Tan; Jie Chen; Qifeng Liu; Yike Guo",
        "year": "2023",
        "venue": "ACM MM",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2023_ACMMM_CoMoSpeech_Consistency_Speech_Synthesis.pdf",
        "url": "https://arxiv.org/abs/2305.06908",
        "pdf_urls": ["https://arxiv.org/pdf/2305.06908.pdf"],
        "relevance": "Consistency-model speech synthesis reference for low-step generation from acoustic conditions.",
    },
    {
        "key": "yang_uniaudio_2024",
        "title": "UniAudio: Towards Universal Audio Generation with Large Language Models",
        "authors": "Dongchao Yang; Jinchuan Tian; Xu Tan; Rongjie Huang; Songxiang Liu; Haohan Guo; Xuankai Chang; Jiatong Shi; Sheng Zhao; Jiang Bian; Zhou Zhao; Xixin Wu; Helen Meng",
        "year": "2024",
        "venue": "ICML",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2024_ICML_UniAudio_Universal_Audio_Generation.pdf",
        "url": "https://proceedings.mlr.press/v235/yang24x.html",
        "pdf_urls": [
            "https://raw.githubusercontent.com/mlresearch/v235/main/assets/yang24x/yang24x.pdf",
            "https://proceedings.mlr.press/v235/yang24x/yang24x.pdf",
        ],
        "relevance": "Universal audio-token generation reference for downstream audio decoder design.",
    },
    {
        "key": "yang_uniaudio15_2024",
        "title": "UniAudio 1.5: Large Language Model-driven Audio Codec is A Few-shot Audio Task Learner",
        "authors": "Dongchao Yang; Haohan Guo; Yuanyuan Wang; Rongjie Huang; Xiang Li; Xu Tan; Xixin Wu; Helen Meng",
        "year": "2024",
        "venue": "NeurIPS",
        "ccf_a": "Yes",
        "group": "P1 generative speech decoder",
        "filename": "2024_NeurIPS_UniAudio_1_5_LLM_Driven_Audio_Codec.pdf",
        "url": "https://arxiv.org/abs/2406.10056",
        "pdf_urls": [
            "https://proceedings.neurips.cc/paper_files/paper/2024/file/6801fa3fd290229efc490ee0cf1c5687-Paper-Conference.pdf",
            "https://arxiv.org/pdf/2406.10056.pdf",
        ],
        "relevance": "LLM-driven audio codec reference for treating audio tokens as a language interface.",
    },
    {
        "key": "borsos_audiolm_2022",
        "title": "AudioLM: a Language Modeling Approach to Audio Generation",
        "authors": "Zalan Borsos; Raphael Marinier; Damien Vincent; Eugene Kharitonov; Olivier Pietquin; Matt Sharifi; Dominik Roblek; Olivier Teboul; David Grangier; Marco Tagliasacchi; Neil Zeghidour",
        "year": "2022",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2022_arXiv_AudioLM_Language_Modeling_Audio_Generation.pdf",
        "url": "https://arxiv.org/abs/2209.03143",
        "pdf_urls": ["https://arxiv.org/pdf/2209.03143.pdf"],
        "relevance": "Foundational semantic-to-acoustic token LM; useful for EEG-derived token completion.",
    },
    {
        "key": "borsos_soundstorm_2023",
        "title": "SoundStorm: Efficient Parallel Audio Generation",
        "authors": "Zalan Borsos; Matt Sharifi; Damien Vincent; Eugene Kharitonov; Neil Zeghidour; Marco Tagliasacchi",
        "year": "2023",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2023_arXiv_SoundStorm_Parallel_Audio_Generation.pdf",
        "url": "https://arxiv.org/abs/2305.09636",
        "pdf_urls": ["https://arxiv.org/pdf/2305.09636.pdf"],
        "relevance": "Parallel codec-token generation reference for faster decoder completion.",
    },
    {
        "key": "wang_valle_2023",
        "title": "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers",
        "authors": "Chengyi Wang; Sanyuan Chen; Yu Wu; Ziqiang Zhang; Long Zhou; Shujie Liu; Zhuo Chen; Yanqing Liu; Huaming Wang; Jinyu Li; Lei He; Sheng Zhao; Furu Wei",
        "year": "2023",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2023_arXiv_VALL_E_Neural_Codec_Language_Model.pdf",
        "url": "https://arxiv.org/abs/2301.02111",
        "pdf_urls": ["https://arxiv.org/pdf/2301.02111.pdf"],
        "relevance": "Canonical neural codec LM for prompt-based voice generation.",
    },
    {
        "key": "chen_valle2_2024",
        "title": "VALL-E 2: Neural Codec Language Models are Human Parity Zero-Shot Text to Speech Synthesizers",
        "authors": "Sanyuan Chen; Shujie Liu; Long Zhou; Yanqing Liu; Xu Tan; Jinyu Li; Sheng Zhao; Yao Qian; Furu Wei",
        "year": "2024",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2024_arXiv_VALL_E_2_Human_Parity_Zero_Shot_TTS.pdf",
        "url": "https://arxiv.org/abs/2406.05370",
        "pdf_urls": ["https://arxiv.org/pdf/2406.05370.pdf"],
        "relevance": "Improved grouped codec modeling reference for long sequence stability.",
    },
    {
        "key": "defossez_encodec_2022",
        "title": "High Fidelity Neural Audio Compression",
        "authors": "Alexandre Defossez; Jade Copet; Gabriel Synnaeve; Yossi Adi",
        "year": "2022",
        "venue": "OpenReview / arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2022_arXiv_EnCodec_High_Fidelity_Neural_Audio_Compression.pdf",
        "url": "https://arxiv.org/abs/2210.13438",
        "pdf_urls": ["https://arxiv.org/pdf/2210.13438.pdf"],
        "relevance": "Practical neural codec backend used by VALL-E/VoiceCraft-style token decoders.",
    },
    {
        "key": "zhang_speechtokenizer_2023",
        "title": "SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models",
        "authors": "Xin Zhang; Dong Zhang; Shimin Li; Yaqian Zhou; Xipeng Qiu",
        "year": "2023",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2023_arXiv_SpeechTokenizer_Unified_Speech_Tokenizer.pdf",
        "url": "https://arxiv.org/abs/2308.16692",
        "pdf_urls": ["https://arxiv.org/pdf/2308.16692.pdf"],
        "relevance": "Hierarchical semantic/acoustic speech tokenization; close to grouped EEG-token alignment.",
    },
    {
        "key": "wang_maskgct_2024",
        "title": "MaskGCT: Zero-Shot Text-to-Speech with Masked Generative Codec Transformer",
        "authors": "Yuancheng Wang; Haoyue Zhan; Liwei Liu; Ruihong Zeng; Haotian Guo; Jiachen Zheng; Qiang Zhang; Xueyao Zhang; Shunsi Zhang; Zhizheng Wu",
        "year": "2024",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2024_arXiv_MaskGCT_Masked_Generative_Codec_Transformer.pdf",
        "url": "https://arxiv.org/abs/2409.00750",
        "pdf_urls": ["https://arxiv.org/pdf/2409.00750.pdf"],
        "relevance": "Two-stage semantic-to-acoustic token generation reference for incomplete EEG-derived conditions.",
    },
    {
        "key": "kyutai_moshi_2024",
        "title": "Moshi/Mimi real-time speech foundation model",
        "authors": "Kyutai research team",
        "year": "2024",
        "venue": "arXiv",
        "ccf_a": "No",
        "group": "P2 background/foundation",
        "filename": "2024_arXiv_Moshi_Mimi_Real_Time_Speech_Foundation_Model.pdf",
        "url": "https://arxiv.org/abs/2410.00037",
        "pdf_urls": ["https://arxiv.org/pdf/2410.00037.pdf"],
        "relevance": "Streaming speech-text/audio-token foundation reference for future real-time EEG voice decoding.",
    },
]


def download(url: str, dest: Path) -> tuple[bool, str]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 paper-reference-collector/1.0",
            "Accept": "application/pdf,text/html;q=0.8,*/*;q=0.5",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            content = response.read()
    except Exception as exc:
        return False, f"{url} -> {exc}"

    if b"%PDF" not in content[:2048] or len(content) < 20_000:
        return False, f"{url} -> response is not a valid-looking PDF ({len(content)} bytes)"

    dest.write_bytes(content)
    return True, f"{url} -> ok ({len(content)} bytes)"


def escape_bibtex(value: str) -> str:
    return value.replace("&", "\\&").replace("_", "\\_")


def bibtex_entry(paper: dict, local_path: str) -> str:
    authors = " and ".join(part.strip() for part in paper["authors"].split(";"))
    entry_type = "inproceedings" if paper["ccf_a"] == "Yes" else "misc"
    return textwrap.dedent(
        f"""\
        @{entry_type}{{{paper['key']},
          title = {{{escape_bibtex(paper['title'])}}},
          author = {{{escape_bibtex(authors)}}},
          year = {{{paper['year']}}},
          note = {{{escape_bibtex(paper['venue'])}; CCF-A={paper['ccf_a']}}},
          url = {{{paper['url']}}},
          file = {{{local_path}:application/pdf}},
        }}
        """
    )


def write_csv(rows: list[dict]) -> None:
    fields = [
        "key",
        "title",
        "authors",
        "year",
        "venue",
        "ccf_a",
        "group",
        "url",
        "local_pdf_path",
        "download_status",
        "relevance",
    ]
    with (ROOT / "papers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_readme(rows: list[dict]) -> None:
    groups = [
        "P0 direct EEG-to-speech",
        "P0 audio codec/token target",
        "P1 generative speech decoder",
        "P2 background/foundation",
    ]
    lines = [
        "# EEG-to-Speech CCF-A + Key arXiv Paper References",
        "",
        "This folder collects papers for the current EEG-to-speech / waveform reconstruction work in this repository.",
        "The selection prioritizes CCF-A venues and keeps a small number of direct EEG-to-speech or audio-token foundation papers that are essential even when they are arXiv-only.",
        "",
        "## Files",
        "",
        "- `pdf/`: local PDF copies.",
        "- `papers.csv`: structured index with venue/status, CCF-A flag, URL, local PDF path, and relevance note.",
        "- `papers.bib`: BibTeX entries pointing at the local PDF files.",
        "- `metadata/download_log.txt`: download and local-reuse log.",
        "- `metadata/search_queries.md`: search directions, inclusion rule, and PDF source preference.",
        "",
        "## Reading Order",
        "",
    ]
    for group in groups:
        lines.extend([f"### {group}", "", "| Year | Venue | CCF-A | Paper | Why it matters |", "| --- | --- | --- | --- | --- |"])
        for paper in [row for row in rows if row["group"] == group]:
            path = Path(paper["local_pdf_path"])
            rel = path.relative_to(ROOT) if path.exists() else path
            title = paper["title"].replace("|", "\\|")
            relevance = paper["relevance"].replace("|", "\\|")
            lines.append(
                f"| {paper['year']} | {paper['venue']} | {paper['ccf_a']} | [{title}]({rel.as_posix()}) | {relevance} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Notes",
            "",
            "- `CCF-A=Yes` means the paper is in the planned CCF-A core set or attached to a CCF-A venue/status used by the project reading list.",
            "- Direct EEG-to-speech papers are retained even when arXiv-only because they are the closest methodological references for KaraOne/FEIS reconstruction.",
            "- If this ignored folder needs to be committed, use `git add -f paper-ref/eeg-to-speech-ccf-a`.",
            "",
        ]
    )
    (ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    log_lines: list[str] = [f"Download run: {time.strftime('%Y-%m-%d %H:%M:%S')}"]

    for paper in PAPERS:
        dest = PDF_DIR / paper["filename"]
        status = "missing"
        local_source = paper.get("local_source")
        if local_source and Path(local_source).exists():
            shutil.copy2(local_source, dest)
            status = f"reused local source: {local_source}"
            log_lines.append(f"[reuse] {paper['key']}: {local_source} -> {dest}")
        elif dest.exists() and b"%PDF" in dest.read_bytes()[:2048]:
            status = f"already exists; source candidates: {'; '.join(paper['pdf_urls'])}"
            log_lines.append(f"[skip] {paper['key']}: {dest} already exists; source candidates: {'; '.join(paper['pdf_urls'])}")
        else:
            for url in paper["pdf_urls"]:
                ok, message = download(url, dest)
                log_lines.append(f"[download] {paper['key']}: {message}")
                if ok:
                    status = f"downloaded: {url}"
                    break
            if status == "missing" and dest.exists():
                dest.unlink()

        row = {k: v for k, v in paper.items() if k not in {"pdf_urls", "local_source"}}
        row["local_pdf_path"] = str(dest)
        row["download_status"] = status
        rows.append(row)

    write_csv(rows)
    write_search_queries()
    (ROOT / "papers.bib").write_text(
        "\n".join(bibtex_entry(row, str(Path(row["local_pdf_path"]).relative_to(ROOT))) for row in rows),
        encoding="utf-8",
    )
    write_readme(rows)
    (METADATA_DIR / "download_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    missing = [row for row in rows if not Path(row["local_pdf_path"]).exists()]
    if missing:
        print("Missing PDFs:")
        for row in missing:
            print(f"- {row['key']}: {row['title']}")
        raise SystemExit(1)

    print(f"Wrote {len(rows)} paper records to {ROOT}")
    print(f"PDF count: {len(list(PDF_DIR.glob('*.pdf')))}")


def write_search_queries() -> None:
    lines = [
        "# Search Queries and Selection Notes",
        "",
        "## User-requested search directions",
        "",
        "- EEG speech reconstruction",
        "- EEG-to-speech reconstruction diffusion",
        "- neural audio codec speech generation",
        "- speech token language model audio generation",
        "- factorized codec speech synthesis",
        "",
        "## Inclusion rule",
        "",
        "- Include CCF-A venue papers that define audio representation, codec/tokenization, or speech generation interfaces relevant to EEG-to-wav reconstruction.",
        "- Include direct EEG-to-speech / neural decoding papers even when arXiv-only because they are closest to the FEIS/KaraOne task.",
        "- Keep the set compact: about 25 papers, grouped by implementation utility rather than exhaustive citation coverage.",
        "",
        "## PDF source preference",
        "",
        "1. Reuse existing local PDFs under `paper-ref/unclassified-root-papers/` when present.",
        "2. Prefer official proceedings PDFs for CCF-A papers.",
        "3. Fall back to arXiv PDFs when proceedings PDFs are unavailable or unstable.",
        "4. Record the stable landing page in `papers.csv:url` and the actual source candidates in `download_status`.",
        "",
    ]
    (METADATA_DIR / "search_queries.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
