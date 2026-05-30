#!/usr/bin/env python3
"""Build a step-by-step exploration notebook for ds006104."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "notebooks" / "explore_ds006104_step_by_step.ipynb"


def md(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": dedent(text).strip("\n"),
    }


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip("\n"),
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(
        md(
            """
            # Explore `ds006104` Step by Step

            这个 notebook 面向你当前已经下载到本地的 `ds006104` 数据集，目标是按下面的顺序逐步探究：

            1. 读取数据集级文档与 BIDS 元数据
            2. 清点被试、session、task、文件完整性
            3. 汇总事件表，理解刺激与 TMS 设计
            4. 读取一个本地可用的原始 EDF，查看通道与时长
            5. 基于 `stimulus` 事件切 epoch，做基础 ERP / PSD / 通道统计

            说明：你当前这份 datalad/git-annex 数据里，部分 `*_eeg.edf` 目标文件并不在本地，但事件表、channels 和 sidecar JSON 是完整的。所以 notebook 会：

            - 先完整跑元数据与事件探索
            - 自动寻找一个本地实际存在的 EDF 做 EEG 演示
            - 如果某些 EDF 缺失，会给出提示而不是报错中断
            """
        )
    )

    cells.append(
        code(
            """
            from __future__ import annotations

            import json
            from pathlib import Path

            import matplotlib.pyplot as plt
            import mne
            import numpy as np
            import pandas as pd
            from IPython.display import Markdown, display

            plt.style.use("seaborn-v0_8-whitegrid")
            pd.set_option("display.max_columns", 50)
            pd.set_option("display.width", 160)
            pd.set_option("display.max_colwidth", 120)

            DATASET_ROOT = Path("data/raw/openneuro/ds006104_datalad")
            OUTPUT_ROOT = Path("outputs") / "ds006104_notebook_outputs"
            OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

            assert DATASET_ROOT.exists(), f"Dataset root not found: {DATASET_ROOT}"
            print(f"Dataset root: {DATASET_ROOT.resolve()}")
            print(f"Notebook outputs: {OUTPUT_ROOT.resolve()}")
            """
        )
    )

    cells.append(
        md(
            """
            ## 1. 读取数据集说明文档

            先看数据集自己的 `README`、`dataset_description.json`、`participants.tsv`。这一步的目标是先建立整体认知，再进入文件级分析。
            """
        )
    )

    cells.append(
        code(
            """
            readme_text = (DATASET_ROOT / "README").read_text(encoding="utf-8")
            dataset_description = json.loads(
                (DATASET_ROOT / "dataset_description.json").read_text(encoding="utf-8")
            )
            participants = pd.read_csv(DATASET_ROOT / "participants.tsv", sep="\\t")
            participants_json = json.loads(
                (DATASET_ROOT / "participants.json").read_text(encoding="utf-8")
            )
            changes_text = (DATASET_ROOT / "CHANGES").read_text(encoding="utf-8")

            print("README preview:")
            print("-" * 80)
            print("\\n".join(readme_text.splitlines()[:35]))
            print("-" * 80)

            summary = {
                "Name": dataset_description.get("Name"),
                "BIDSVersion": dataset_description.get("BIDSVersion"),
                "DatasetType": dataset_description.get("DatasetType"),
                "DatasetDOI": dataset_description.get("DatasetDOI"),
                "License": dataset_description.get("License"),
                "Authors_count": len(dataset_description.get("Authors", [])),
            }
            print(summary)
            print("\\nParticipants table shape:", participants.shape)
            participants.head()
            """
        )
    )

    cells.append(
        code(
            """
            display(Markdown("### Dataset Description"))
            pd.DataFrame(
                [
                    {
                        "field": key,
                        "value": value if not isinstance(value, list) else "; ".join(map(str, value)),
                    }
                    for key, value in dataset_description.items()
                ]
            ).head(20)
            """
        )
    )

    cells.append(
        code(
            """
            display(Markdown("### Participants Columns Description"))
            pd.DataFrame(
                [
                    {"column": key, "description": value.get("Description", "")}
                    for key, value in participants_json.items()
                ]
            )
            """
        )
    )

    cells.append(
        code(
            """
            print("CHANGES:")
            print(changes_text)
            """
        )
    )

    cells.append(
        md(
            """
            ## 2. 清点 BIDS 结构和文件完整性

            这里我们把所有与 EEG 分析直接相关的文件扫一遍：

            - `*_eeg.edf`
            - `*_events.tsv`
            - `*_channels.tsv`
            - `*_eeg.json`
            - `*_coordsystem.json`

            同时确认哪些 EDF 在本地是“链接存在但目标文件缺失”的状态。
            """
        )
    )

    cells.append(
        code(
            """
            def parse_bids_entities(path: Path) -> dict[str, str]:
                entities = {}
                stem = path.name
                for suffix in [".edf", ".tsv", ".json"]:
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                for part in stem.split("_"):
                    if "-" in part:
                        key, value = part.split("-", 1)
                        entities[key] = value
                return entities


            edf_files = sorted(DATASET_ROOT.rglob("*_eeg.edf"))
            events_files = sorted(DATASET_ROOT.rglob("*_events.tsv"))
            channels_files = sorted(DATASET_ROOT.rglob("*_channels.tsv"))
            eeg_json_files = sorted(DATASET_ROOT.rglob("*_eeg.json"))
            coordsystem_files = sorted(DATASET_ROOT.rglob("*_coordsystem.json"))

            file_inventory = pd.DataFrame(
                [
                    {"kind": "edf", "count": len(edf_files), "existing_targets": sum(p.exists() for p in edf_files)},
                    {
                        "kind": "events_tsv",
                        "count": len(events_files),
                        "existing_targets": sum(p.exists() for p in events_files),
                    },
                    {
                        "kind": "channels_tsv",
                        "count": len(channels_files),
                        "existing_targets": sum(p.exists() for p in channels_files),
                    },
                    {
                        "kind": "eeg_json",
                        "count": len(eeg_json_files),
                        "existing_targets": sum(p.exists() for p in eeg_json_files),
                    },
                    {
                        "kind": "coordsystem_json",
                        "count": len(coordsystem_files),
                        "existing_targets": sum(p.exists() for p in coordsystem_files),
                    },
                ]
            )

            file_inventory
            """
        )
    )

    cells.append(
        code(
            """
            broken_edf = [p for p in edf_files if p.is_symlink() and not p.exists()]
            existing_edf = [p for p in edf_files if p.exists()]

            print(f"Total EDF links: {len(edf_files)}")
            print(f"Existing EDF targets: {len(existing_edf)}")
            print(f"Broken EDF symlinks: {len(broken_edf)}")
            print("\\nFirst broken examples:")
            for p in broken_edf[:8]:
                print("-", p)
            """
        )
    )

    cells.append(
        code(
            """
            records = []
            for p in edf_files:
                ent = parse_bids_entities(p)
                records.append(
                    {
                        "subject": f"sub-{ent.get('sub')}" if ent.get("sub") else None,
                        "session": f"ses-{ent.get('ses')}" if ent.get("ses") else None,
                        "task": ent.get("task"),
                        "edf_exists": p.exists(),
                        "path": str(p),
                    }
                )

            edf_index = pd.DataFrame(records).sort_values(["subject", "session", "task"]).reset_index(drop=True)
            edf_index.head(12)
            """
        )
    )

    cells.append(
        code(
            """
            display(Markdown("### EDF availability by task"))
            edf_index.groupby(["session", "task", "edf_exists"]).size().rename("n").reset_index()
            """
        )
    )

    cells.append(
        md(
            """
            ## 3. 被试、study、task 的整体分布

            先确认两期 study 的被试规模，以及它们对应的 task 结构。
            """
        )
    )

    cells.append(
        code(
            """
            participants["study"].value_counts().rename_axis("study").reset_index(name="n_participants")
            """
        )
    )

    cells.append(
        code(
            """
            task_counts = edf_index.groupby(["subject", "session"])["task"].agg(list).reset_index()
            task_counts.head(12)
            """
        )
    )

    cells.append(
        code(
            """
            summary_by_task = (
                edf_index.groupby("task")
                .agg(
                    n_recordings=("path", "size"),
                    n_existing_edf=("edf_exists", "sum"),
                    n_missing_edf=("edf_exists", lambda s: (~s).sum()),
                )
                .reset_index()
                .sort_values("task")
            )
            summary_by_task
            """
        )
    )

    cells.append(
        md(
            """
            ## 4. 读取所有事件表并做统一清洗

            这里做几件事：

            - 读取全部 `events.tsv`
            - 补充 `subject / session / task`
            - 清洗 `phoneme1/2/3` 中的空字符和 `n/a`
            - 区分 `stimulus` 与 `TMS`
            - 生成一个统一的 `stimulus_label`
            """
        )
    )

    cells.append(
        code(
            """
            def clean_token(value):
                if pd.isna(value):
                    return np.nan
                value = str(value).replace(chr(0), "").strip()
                if value == "" or value.lower() == "n/a":
                    return np.nan
                return value


            all_events = []
            for path in events_files:
                ent = parse_bids_entities(path)
                df = pd.read_csv(path, sep="\\t")
                df["subject"] = f"sub-{ent.get('sub')}" if ent.get("sub") else None
                df["session"] = f"ses-{ent.get('ses')}" if ent.get("ses") else None
                df["task"] = ent.get("task")
                df["events_path"] = str(path)
                for col in [
                    "phoneme1",
                    "phoneme2",
                    "phoneme3",
                    "category",
                    "manner",
                    "place",
                    "voicing",
                    "tms_target",
                ]:
                    if col in df.columns:
                        df[col] = df[col].map(clean_token)
                all_events.append(df)

            all_events = pd.concat(all_events, ignore_index=True)
            all_events["trial_type"] = all_events["trial_type"].map(clean_token)
            all_events["stimulus_label"] = (
                all_events[["phoneme1", "phoneme2", "phoneme3"]]
                .fillna("")
                .agg("".join, axis=1)
                .replace("", np.nan)
            )

            print(all_events.shape)
            all_events.head()
            """
        )
    )

    cells.append(
        code(
            """
            all_events.groupby(["task", "trial_type"]).size().rename("n_events").reset_index()
            """
        )
    )

    cells.append(
        code(
            """
            stim_events = all_events.query("trial_type == 'stimulus'").copy()
            tms_events = all_events.query("trial_type == 'TMS'").copy()

            print("Stimulus rows:", len(stim_events))
            print("TMS rows:", len(tms_events))
            """
        )
    )

    cells.append(
        md(
            """
            ## 5. 任务级事件统计

            这一步重点回答几个问题：

            - 每个 task 有多少刺激事件？
            - 不同 task 的 `tms_target` 怎么分布？
            - `stimulus_label` 的词表长什么样？
            - `singlephoneme` / `phonemes` / `Words` 的类别字段含义是什么？
            """
        )
    )

    cells.append(
        code(
            """
            stim_events.groupby("task").size().rename("n_stimulus_events").reset_index().sort_values("task")
            """
        )
    )

    cells.append(
        code(
            """
            (
                stim_events.groupby(["task", "tms_target"])
                .size()
                .rename("n")
                .reset_index()
                .sort_values(["task", "n"], ascending=[True, False])
                .head(30)
            )
            """
        )
    )

    cells.append(
        code(
            """
            (
                stim_events.groupby(["task", "category"])
                .size()
                .rename("n")
                .reset_index()
                .sort_values(["task", "n"], ascending=[True, False])
                .head(30)
            )
            """
        )
    )

    cells.append(
        code(
            """
            for task_name in sorted(stim_events["task"].dropna().unique()):
                labels = stim_events.loc[stim_events["task"] == task_name, "stimulus_label"].dropna()
                print(f"\\nTask: {task_name}")
                print(f"Unique stimulus labels: {labels.nunique()}")
                print(labels.value_counts().head(20))
            """
        )
    )

    cells.append(
        code(
            """
            trial_summary = (
                all_events.groupby(["subject", "session", "task", "trial_type"])
                .size()
                .rename("n_events")
                .reset_index()
                .pivot_table(
                    index=["subject", "session", "task"],
                    columns="trial_type",
                    values="n_events",
                    fill_value=0,
                )
                .reset_index()
            )
            trial_summary.head(20)
            """
        )
    )

    cells.append(
        md(
            """
            ## 6. 查看 sidecar JSON 和 channels 信息

            先抽一条 recording，看原始采样率、参考电极、设备和通道布局。
            """
        )
    )

    cells.append(
        code(
            """
            sample_json_path = DATASET_ROOT / "sub-S01" / "ses-02" / "eeg" / "sub-S01_ses-02_task-singlephoneme_eeg.json"
            sample_channels_path = DATASET_ROOT / "sub-S01" / "ses-02" / "eeg" / "sub-S01_ses-02_task-singlephoneme_channels.tsv"
            sample_coordsystem_path = DATASET_ROOT / "sub-S01" / "ses-02" / "eeg" / "sub-S01_ses-02_coordsystem.json"

            sample_eeg_json = json.loads(sample_json_path.read_text(encoding="utf-8"))
            sample_channels = pd.read_csv(sample_channels_path, sep="\\t")
            sample_coordsystem = json.loads(sample_coordsystem_path.read_text(encoding="utf-8"))

            pd.DataFrame(
                [
                    {
                        "field": k,
                        "value": v if not isinstance(v, dict) else json.dumps(v, ensure_ascii=False),
                    }
                    for k, v in sample_eeg_json.items()
                ]
            )
            """
        )
    )

    cells.append(code("""sample_channels.head(10)"""))
    cells.append(code("""sample_coordsystem"""))

    cells.append(
        md(
            """
            ## 7. 选择一个本地存在的 EDF 进行原始 EEG 探索

            由于你当前本地有 8 个 EDF 还没把 annex 实体拉下来，这里自动选择一个真正存在的 EDF。优先顺序是：

            1. `singlephoneme`
            2. `phonemes`
            3. `Words`

            这样可以尽快进入 epoch 和 ERP 分析。
            """
        )
    )

    cells.append(
        code(
            """
            def choose_existing_edf(index_df: pd.DataFrame) -> Path:
                priority = {"singlephoneme": 0, "phonemes": 1, "Words": 2}
                candidates = index_df[index_df["edf_exists"]].copy()
                if candidates.empty:
                    raise RuntimeError("No local EDF payloads available.")
                candidates["task_priority"] = candidates["task"].map(lambda x: priority.get(x, 99))
                row = candidates.sort_values(["task_priority", "subject", "session"]).iloc[0]
                return Path(row["path"])


            selected_edf = choose_existing_edf(edf_index)
            selected_entities = parse_bids_entities(selected_edf)
            selected_events_tsv = selected_edf.with_name(selected_edf.name.replace("_eeg.edf", "_events.tsv"))
            selected_channels_tsv = selected_edf.with_name(selected_edf.name.replace("_eeg.edf", "_channels.tsv"))
            selected_eeg_json = selected_edf.with_name(selected_edf.name.replace("_eeg.edf", "_eeg.json"))

            print("Selected EDF:", selected_edf)
            print("Selected task:", selected_entities.get("task"))
            print("Events TSV exists:", selected_events_tsv.exists())
            print("Channels TSV exists:", selected_channels_tsv.exists())
            print("EEG JSON exists:", selected_eeg_json.exists())
            """
        )
    )

    cells.append(
        code(
            """
            raw = mne.io.read_raw_edf(selected_edf, preload=False, verbose=False)
            raw
            """
        )
    )

    cells.append(
        code(
            """
            raw_info_summary = {
                "n_channels": raw.info["nchan"],
                "sfreq": raw.info["sfreq"],
                "duration_sec": raw.n_times / raw.info["sfreq"],
                "highpass": raw.info["highpass"],
                "lowpass": raw.info["lowpass"],
                "line_freq": raw.info.get("line_freq"),
                "first_channels": raw.ch_names[:10],
            }
            raw_info_summary
            """
        )
    )

    cells.append(
        code(
            """
            selected_events_df = pd.read_csv(selected_events_tsv, sep="\\t")
            for col in ["phoneme1", "phoneme2", "phoneme3", "category", "manner", "place", "voicing", "tms_target"]:
                if col in selected_events_df.columns:
                    selected_events_df[col] = selected_events_df[col].map(clean_token)
            selected_events_df["stimulus_label"] = (
                selected_events_df[[c for c in ["phoneme1", "phoneme2", "phoneme3"] if c in selected_events_df.columns]]
                .fillna("")
                .agg("".join, axis=1)
                .replace("", np.nan)
            )
            selected_events_df.head()
            """
        )
    )

    cells.append(
        md(
            """
            ## 8. 原始 EEG 的基础可视化

            先看一小段原始波形，再看通道功率谱。
            """
        )
    )

    cells.append(
        code(
            """
            raw_preview = raw.copy().load_data().pick("eeg")
            raw_preview.crop(tmin=0, tmax=min(20, raw_preview.times[-1]))
            raw_preview.plot(scalings="auto", n_channels=min(20, raw_preview.info["nchan"]))
            """
        )
    )

    cells.append(
        code(
            """
            raw_psd = raw.copy().load_data().pick("eeg")
            raw_psd.compute_psd(fmin=0.5, fmax=45.0).plot(average=True)
            """
        )
    )

    cells.append(
        md(
            """
            ## 9. 把 `stimulus` 事件转成 MNE events

            这里我们不依赖 EDF 里内嵌的注释，而是直接用 BIDS `events.tsv` 的 `onset` 来构造事件。这样逻辑更透明，也更适合后续和事件元数据联动。
            """
        )
    )

    cells.append(
        code(
            """
            stim_df = selected_events_df.query("trial_type == 'stimulus'").copy().reset_index(drop=True)
            assert not stim_df.empty, "No stimulus rows found for the selected recording."

            unique_labels = sorted(stim_df["stimulus_label"].dropna().unique())
            event_id = {label: idx + 1 for idx, label in enumerate(unique_labels)}
            stim_df["sample"] = (stim_df["onset"] * raw.info["sfreq"]).round().astype(int)
            stim_df["event_code"] = stim_df["stimulus_label"].map(event_id)

            mne_events = stim_df[["sample", "sample", "event_code"]].copy()
            mne_events.iloc[:, 1] = 0
            mne_events = mne_events.to_numpy(dtype=int)

            print("n stimulus events:", len(stim_df))
            print("n unique stimulus labels:", len(unique_labels))
            print("First 10 event ids:", list(event_id.items())[:10])
            mne_events[:10]
            """
        )
    )

    cells.append(
        md(
            """
            ## 10. 预处理一个基础版本并切 epoch

            这里采用一个尽量保守、适合初探的流程：

            - 选择 EEG 通道
            - 1 到 40 Hz band-pass
            - 不改参考，先保留数据集原始参考信息
            - 以 stimulus onset 为 0 点
            - 截取 `-0.2s ~ 0.8s`
            - 做 baseline correction

            后续你可以再决定是否做重参考、ICA、坏道处理等。
            """
        )
    )

    cells.append(
        code(
            """
            raw_epo = raw.copy().load_data().pick("eeg")
            raw_epo.filter(l_freq=1.0, h_freq=40.0, verbose=False)

            epochs = mne.Epochs(
                raw_epo,
                mne_events,
                event_id=event_id,
                tmin=-0.2,
                tmax=0.8,
                baseline=(-0.2, 0.0),
                preload=True,
                reject_by_annotation=False,
                verbose=False,
            )

            epochs
            """
        )
    )

    cells.append(
        code(
            """
            epoch_metadata = stim_df.copy()
            epoch_metadata = epoch_metadata[
                ["onset", "stimulus_label", "category", "manner", "place", "voicing", "tms_target", "trial"]
            ]
            epochs.metadata = epoch_metadata
            print(epochs)
            epochs.metadata.head()
            """
        )
    )

    cells.append(
        md(
            """
            ## 11. 看整体 ERP 和若干条件平均

            第一层先看总平均；第二层按数据里最自然的标签切开。对于 `singlephoneme`，通常适合看元音 vs 辅音、voiced vs unvoiced、place/manner。
            """
        )
    )

    cells.append(
        code(
            """
            evoked_all = epochs.average()
            evoked_all.plot(spatial_colors=True, gfp=True)
            """
        )
    )

    cells.append(
        code(
            """
            if "category" in epochs.metadata.columns and epochs.metadata["category"].notna().any():
                cat_counts = epochs.metadata["category"].value_counts(dropna=True)
                print(cat_counts)
                keep_categories = [c for c, n in cat_counts.items() if n >= 5]
                if keep_categories:
                    evokeds = {cat: epochs[epochs.metadata["category"] == cat].average() for cat in keep_categories}
                    mne.viz.plot_compare_evokeds(evokeds, combine="mean")
            """
        )
    )

    cells.append(
        code(
            """
            if "voicing" in epochs.metadata.columns and epochs.metadata["voicing"].notna().any():
                voice_counts = epochs.metadata["voicing"].value_counts(dropna=True)
                print(voice_counts)
                keep_voicing = [c for c, n in voice_counts.items() if n >= 5]
                if keep_voicing:
                    evokeds = {label: epochs[epochs.metadata["voicing"] == label].average() for label in keep_voicing}
                    mne.viz.plot_compare_evokeds(evokeds, combine="mean")
            """
        )
    )

    cells.append(
        code(
            """
            if "place" in epochs.metadata.columns and epochs.metadata["place"].notna().any():
                place_counts = epochs.metadata["place"].value_counts(dropna=True)
                print(place_counts)
                keep_place = [c for c, n in place_counts.items() if n >= 5]
                if keep_place:
                    evokeds = {label: epochs[epochs.metadata["place"] == label].average() for label in keep_place}
                    mne.viz.plot_compare_evokeds(evokeds, combine="mean")
            """
        )
    )

    cells.append(
        md(
            """
            ## 12. 看几个代表时间窗的 topomap

            这一步帮助我们快速判断刺激后不同时段的空间分布。
            """
        )
    )

    cells.append(
        code(
            """
            times = np.array([0.05, 0.10, 0.15, 0.20, 0.30, 0.40])
            evoked_all.plot_topomap(times=times, ch_type="eeg")
            """
        )
    )

    cells.append(
        md(
            """
            ## 13. 单试次幅度统计

            这里给出一个非常适合初探的 summary：

            - 每个 epoch 在若干时间窗里的全脑平均绝对振幅
            - 再按条件做 boxplot / 均值比较
            """
        )
    )

    cells.append(
        code(
            """
            epoch_data = epochs.get_data(copy=True)
            times = epochs.times

            windows = {
                "pre_baseline": (-0.2, 0.0),
                "early_50_150ms": (0.05, 0.15),
                "mid_150_300ms": (0.15, 0.30),
                "late_300_500ms": (0.30, 0.50),
            }

            summary_df = epochs.metadata.copy()
            for name, (t0, t1) in windows.items():
                mask = (times >= t0) & (times <= t1)
                summary_df[name] = np.mean(np.abs(epoch_data[:, :, mask]), axis=(1, 2))

            summary_df.head()
            """
        )
    )

    cells.append(
        code(
            """
            plot_cols = [
                c
                for c in ["category", "voicing", "place", "tms_target"]
                if c in summary_df.columns and summary_df[c].notna().any()
            ]
            metric = "mid_150_300ms"

            for col in plot_cols[:3]:
                fig, ax = plt.subplots(figsize=(8, 4))
                grouped = summary_df.dropna(subset=[col]).groupby(col)[metric]
                labels = []
                values = []
                for label, vals in grouped:
                    if len(vals) >= 5:
                        labels.append(label)
                        values.append(vals.values)
                if values:
                    ax.boxplot(values, tick_labels=labels)
                    ax.set_title(f"{metric} by {col}")
                    ax.set_ylabel("Mean abs amplitude")
                    plt.xticks(rotation=30, ha="right")
                    plt.show()
            """
        )
    )

    cells.append(
        md(
            """
            ## 14. 生成 recording 级元数据总表

            这是后续你做系统性分析时最实用的中间结果之一：把每个 recording 的关键 sidecar 信息、事件数量、EDF 是否在本地等信息统一成一张表。
            """
        )
    )

    cells.append(
        code(
            """
            recording_rows = []
            for edf_path in edf_files:
                ent = parse_bids_entities(edf_path)
                subj = f"sub-{ent.get('sub')}"
                sess = f"ses-{ent.get('ses')}"
                task = ent.get("task")
                eeg_json_path = edf_path.with_name(edf_path.name.replace("_eeg.edf", "_eeg.json"))
                events_path = edf_path.with_name(edf_path.name.replace("_eeg.edf", "_events.tsv"))
                channels_path = edf_path.with_name(edf_path.name.replace("_eeg.edf", "_channels.tsv"))

                eeg_meta = json.loads(eeg_json_path.read_text(encoding="utf-8")) if eeg_json_path.exists() else {}
                ev = pd.read_csv(events_path, sep="\\t") if events_path.exists() else pd.DataFrame()
                stim = ev[ev["trial_type"] == "stimulus"] if "trial_type" in ev.columns else pd.DataFrame()
                tms = ev[ev["trial_type"] == "TMS"] if "trial_type" in ev.columns else pd.DataFrame()
                ch = pd.read_csv(channels_path, sep="\\t") if channels_path.exists() else pd.DataFrame()

                unique_targets = ""
                if not stim.empty and "tms_target" in stim.columns:
                    cleaned_targets = sorted({str(x) for x in stim["tms_target"].dropna().unique()})
                    unique_targets = ", ".join(cleaned_targets)

                recording_rows.append(
                    {
                        "subject": subj,
                        "session": sess,
                        "task": task,
                        "edf_exists_locally": edf_path.exists(),
                        "sampling_frequency": eeg_meta.get("SamplingFrequency"),
                        "reference": eeg_meta.get("EEGReference"),
                        "ground": eeg_meta.get("EEGGround"),
                        "cap_model": eeg_meta.get("CapManufacturersModelName"),
                        "device_model": eeg_meta.get("ManufacturersModelName"),
                        "n_channels_tsv": len(ch),
                        "n_stimulus_events": len(stim),
                        "n_tms_events": len(tms),
                        "unique_tms_targets": unique_targets,
                        "events_path": str(events_path),
                        "edf_path": str(edf_path),
                    }
                )

            recording_summary = (
                pd.DataFrame(recording_rows).sort_values(["subject", "session", "task"]).reset_index(drop=True)
            )
            recording_summary.head(20)
            """
        )
    )

    cells.append(
        code(
            """
            summary_csv = OUTPUT_ROOT / "ds006104_recording_summary.csv"
            recording_summary.to_csv(summary_csv, index=False)
            print(f"Saved: {summary_csv.resolve()}")
            """
        )
    )

    cells.append(
        md(
            """
            ## 15. 下一步建议

            如果你准备继续深入，我建议按这个顺序走：

            1. 先补齐缺失的 8 个 EDF annex 对象，这样 Study 1 的 `sub-P01` 到 `sub-P08` 也能进原始 EEG 分析
            2. 对每个 task 分开做 epoching，因为 `singlephoneme / phonemes / Words` 的刺激结构不同
            3. 明确要预测的标签层级：
               - `singlephoneme`: 单音素类别、元音/辅音、voicing、place、manner
               - `phonemes`: CV / VC 组合、共发音结构、TMS target
               - `Words`: real vs nonce、CVC 序列、TMS target
            4. 再进入机器学习前，先做坏道、重参考、ICA/伪迹检查和 trial-level QC

            如果你愿意，下一步我可以继续直接给你补：

            - 一个 `scripts/` 下可复用的 `.py` 版分析脚本
            - 一个批量遍历所有 recording 的 ERP/PSD 导出脚本
            - 一个面向解码任务的数据准备 notebook
            """
        )
    )

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    notebook = build_notebook()
    OUT_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
