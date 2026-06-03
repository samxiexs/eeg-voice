#!/usr/bin/env python3
"""Generate executed notebooks and Markdown reports for FEIS and KaraOne."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

from eeg_dataset_analysis_lib import (
    analyze_feis,
    analyze_karaone,
    save_bundle_json,
    write_report_md,
)


ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = ROOT / "reports" / "dataset_feasibility"


def notebook_bootstrap_code() -> str:
    return (
        "from pathlib import Path\n"
        "import sys\n"
        "import pandas as pd\n"
        "from IPython.display import Image, Markdown, display\n"
        "\n"
        "def find_project_root(start: Path) -> Path:\n"
        "    for candidate in [start, *start.parents]:\n"
        "        if (candidate / 'scripts' / 'eeg_dataset_analysis_lib.py').exists():\n"
        "            return candidate\n"
        "    raise RuntimeError('Could not locate project root containing scripts/eeg_dataset_analysis_lib.py')\n"
        "\n"
        "PROJECT_ROOT = find_project_root(Path.cwd())\n"
        "sys.path.insert(0, str(PROJECT_ROOT / 'scripts'))\n"
    )


def markdown_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_markdown_cell(text)


def code_cell(text: str) -> nbformat.NotebookNode:
    return nbformat.v4.new_code_cell(text)


def make_feis_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown_cell(
            "# Dataset 1 Analysis: FEIS\n\n"
            "这个 Notebook 直接分析本机下载的 FEIS 数据，重点回答它是否适合后续 EEG 语音解码与语音重建研究。"
        ),
        code_cell(notebook_bootstrap_code() + "from eeg_dataset_analysis_lib import analyze_feis\n"),
        code_cell(
            "bundle = analyze_feis()\n"
            "summary = bundle.summary\n"
            "assets = bundle.assets\n"
            "summary['dataset_name'], summary['subject_folder_count']"
        ),
        markdown_cell("## 一、数据集整体概览"),
        code_cell(
            "overview_df = pd.DataFrame(summary['overview_rows'], columns=['subject', 'trial_count', 'label_count', 'has_full_eeg'])\n"
            "display(overview_df)\n"
            "display(Markdown(f\"- irregular subjects: `{summary['irregular_subjects']}`\"))\n"
            "display(Markdown(f\"- representative subject: `{summary['representative_subject']}`\"))\n"
            "display(Markdown(f\"- channels: `{', '.join(summary['channel_names'])}`\"))\n"
        ),
        code_cell("display(Image(filename=assets['trial_counts']))"),
        markdown_cell("## 二、实验范式分析"),
        code_cell(
            "stage_counts_df = pd.DataFrame(list(summary['representative_stage_counts'].items()), columns=['stage', 'samples'])\n"
            "stage_counts_df['approx_seconds'] = stage_counts_df['samples'] / 256.0\n"
            "display(stage_counts_df)\n"
            "display(Markdown('从 `full_eeg.csv` 的 Stage 列可以直接恢复 trial 顺序：`stimuli -> articulators -> thinking -> speaking -> resting`。'))\n"
        ),
        markdown_cell("## 三、单受试者深度分析"),
        code_cell("display(Image(filename=assets['waveform']))"),
        code_cell("display(Image(filename=assets['channel_std']))"),
        markdown_cell("## 四、事件与标签分析"),
        code_cell(
            "display(Image(filename=assets['labels']))\n"
            "fit_df = pd.DataFrame(summary['research_fit'].items(), columns=['task', 'judgment'])\n"
            "display(fit_df)\n"
        ),
        markdown_cell("## 五、与研究目标的匹配度评估"),
        code_cell(
            "for note in summary['data_quality_notes']:\n"
            "    print('-', note)\n"
            "\n"
            "print('\\nKey conclusion: FEIS is a strong pilot set for imagined phoneme classification, but a weak foundation for speech reconstruction.')\n"
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def make_karaone_notebook() -> nbformat.NotebookNode:
    cells = [
        markdown_cell(
            "# Dataset 2 Analysis: KaraOne\n\n"
            "这个 Notebook 直接分析本机下载的 KaraOne 数据，重点判断它在 EEG 语音解码与语音重建方向上的研究价值。"
        ),
        code_cell(notebook_bootstrap_code() + "from eeg_dataset_analysis_lib import analyze_karaone\n"),
        code_cell(
            "bundle = analyze_karaone()\n"
            "summary = bundle.summary\n"
            "assets = bundle.assets\n"
            "summary['dataset_name'], summary['downloaded_archive_count']"
        ),
        markdown_cell("## 一、数据集整体概览"),
        code_cell(
            "archive_df = pd.DataFrame(summary['archive_rows'], columns=['subject_archive', 'size_gb'])\n"
            "display(archive_df)\n"
            "display(Markdown(f\"- representative subject: `{summary['representative_subject']}`\"))\n"
            "display(Markdown(f\"- sampling rate: `{summary['sampling_rate_hz']}` Hz\"))\n"
            "display(Markdown(f\"- raw duration: `{summary['raw_duration_sec'] / 60:.2f}` minutes\"))\n"
        ),
        code_cell("display(Image(filename=assets['archive_sizes']))"),
        markdown_cell("## 二、实验范式分析"),
        code_cell(
            "interval_df = pd.DataFrame([\n"
            "    ['clearing', summary['clearing_interval_count'], summary['clearing_duration_mean_sec']],\n"
            "    ['thinking', summary['thinking_interval_count'], summary['thinking_duration_mean_sec']],\n"
            "    ['stimulus_like', summary['stimulus_like_interval_count'], summary['stimulus_like_duration_mean_sec']],\n"
            "    ['overt_like', summary['overt_like_interval_count'], summary['overt_like_duration_mean_sec']],\n"
            "], columns=['interval_family', 'count', 'mean_duration_sec'])\n"
            "display(interval_df)\n"
            "display(Markdown('`epoch_inds.mat` 明确给出了 clearing / thinking / speaking-like 区间；结合标签与 wav，可以把 speaking-like 再拆成 stimulus-like 与 overt-like 两段。'))\n"
        ),
        markdown_cell("## 三、单受试者深度分析"),
        code_cell("display(Image(filename=assets['waveform']))"),
        code_cell("display(Image(filename=assets['audio_durations']))"),
        markdown_cell("## 四、事件与标签分析"),
        code_cell(
            "label_df = pd.DataFrame(sorted(summary['representative_label_counter'].items()), columns=['label', 'count'])\n"
            "display(label_df)\n"
            "display(Image(filename=assets['labels']))\n"
            "display(Markdown(f\"Trigger unique values observed via MNE: `{summary['trigger_unique_values']}`\"))\n"
        ),
        markdown_cell("## 五、与研究目标的匹配度评估"),
        code_cell(
            "fit_df = pd.DataFrame(summary['research_fit'].items(), columns=['task', 'judgment'])\n"
            "display(fit_df)\n"
            "for note in summary['data_quality_notes']:\n"
            "    print('-', note)\n"
            "\n"
            "print('\\nKey conclusion: KaraOne is materially better than FEIS for overt-to-imagined transfer and for overt EEG -> acoustic representation studies, but it is still a small-vocabulary dataset.')\n"
        ),
    ]
    return nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def execute_notebook(notebook: nbformat.NotebookNode, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(nbformat.writes(notebook), encoding="utf-8")
    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    ep.preprocess(notebook, {"metadata": {"path": str(ROOT)}})
    output_path.write_text(nbformat.writes(notebook), encoding="utf-8")


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    dataset1_bundle = analyze_feis()
    dataset2_bundle = analyze_karaone()

    save_bundle_json(dataset1_bundle, ROOT / "outputs/dataset_analysis_assets/feis/summary.json")
    save_bundle_json(dataset2_bundle, ROOT / "outputs/dataset_analysis_assets/karaone/summary.json")

    write_report_md(dataset1_bundle, ANALYSIS_DIR / "dataset1_report.md")
    write_report_md(dataset2_bundle, ANALYSIS_DIR / "dataset2_report.md")

    execute_notebook(make_feis_notebook(), ANALYSIS_DIR / "dataset1_analysis.ipynb")
    execute_notebook(make_karaone_notebook(), ANALYSIS_DIR / "dataset2_analysis.ipynb")


if __name__ == "__main__":
    main()
