from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.io import wavfile
from torch.utils.data import DataLoader

BUNDLE_DIR = Path(__file__).resolve().parents[1]
if str(BUNDLE_DIR) not in sys.path:
    sys.path.insert(0, str(BUNDLE_DIR))

from scripts.train_karaone_0711v1 import AudioTargetBank, cache_paths, default_device, load_model, make_eeg_model, move_batch, resolve
from src.audio_features import AudioFeatureConfig, load_codec_backend
from src.karaone_0711v1.data import KaraOne0711Dataset, SplitManifest, make_run_manifest, run_name, write_json
from src.karaone_0711v1.eval import require_flow_gate
from src.karaone_0711v1.model import ConditionalFlowDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EEG-only 0711v1 EnCodec-flow synthesis.")
    parser.add_argument("--config", default=str(BUNDLE_DIR / "configs" / "karaone_0711v1.yaml"))
    parser.add_argument("--encoder", required=True, help="Locked alignment checkpoint")
    parser.add_argument("--flow", required=True, help="Passed-gate flow checkpoint")
    parser.add_argument("--gate", required=True, help="Passed validation gate JSON")
    parser.add_argument("--stage", choices=("overt_like", "thinking"), default=None)
    parser.add_argument("--split", choices=("subject_val", "subject_test"), default="subject_val")
    parser.add_argument("--allow-final-test", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    stage = args.stage or str(cfg["data"]["stage"])
    seed = int(args.seed if args.seed is not None else cfg["run"]["seed"])
    device = torch.device(args.device) if args.device else default_device()
    root = resolve(cfg["data"]["root"])
    manifest = SplitManifest.build(root)
    require_flow_gate(args.gate, manifest)
    if args.split == "subject_test" and not args.allow_final_test:
        raise PermissionError("MM21 synthesis requires --allow-final-test")
    bank = AudioTargetBank(cache_paths(cfg, stage, seed)["audio_targets"])
    if bank.encodec_latent is None:
        raise FileNotFoundError("Adapted target cache has no EnCodec latent array")
    encoder = make_eeg_model(cfg).to(device)
    load_model(args.encoder, encoder, device)
    encoder.eval()
    flow_cfg = cfg["flow"]
    flow = ConditionalFlowDecoder(
        latent_dim=int(bank.encodec_latent.shape[-1]),
        eeg_dim=int(cfg["model"]["d_model"]),
        d_model=int(flow_cfg["d_model"]),
        heads=int(flow_cfg["heads"]),
        layers=int(flow_cfg["layers"]),
    ).to(device)
    load_model(args.flow, flow, device)
    flow.eval()
    dataset = KaraOne0711Dataset(root, args.split, stage, manifest=manifest, eeg_len=int(cfg["data"]["eeg_len"]), sample_rate=int(cfg["data"]["eeg_sample_rate"]))
    data = DataLoader(dataset, batch_size=int(flow_cfg["batch_size"]), shuffle=False, num_workers=0)
    output = resolve(cfg["paths"]["output_root"]) / run_name(stage, "flow", seed) / "wavs" / f"generated_{args.split}"
    output.mkdir(parents=True, exist_ok=True)
    codec_path = resolve(cfg["paths"]["encodec_model"])
    codec = load_codec_backend(AudioFeatureConfig(sample_rate=16000, duration_sec=2.0, target_kind="encodec_latent", backend="encodec_latent", codec_model_name_or_path=str(codec_path), local_files_only=True, codec_bandwidth=6.0))
    saved = []
    with torch.no_grad():
        for batch in data:
            batch = move_batch(batch, device)
            encoded = encoder(batch["eeg"], batch["eeg_valid_len"], batch["topography"])
            latent = flow.sample(encoded["tokens"], encoded["pred_onset_sec"], encoded["pred_duration_sec"], encoded["pred_active_logit"], steps=int(args.steps or flow_cfg["sample_steps"])).cpu().numpy()
            for idx, key in enumerate(batch["key"]):
                audio = codec.decode(latent[idx])
                path = output / f"{str(key).replace(':', '_')}.wav"
                wavfile.write(path, 16000, (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16))
                saved.append(path.name)
                if args.limit and len(saved) >= int(args.limit):
                    break
            if args.limit and len(saved) >= int(args.limit):
                break
    write_json(output / "synthesis_manifest.json", {
        **make_run_manifest(repo_root=BUNDLE_DIR.parent.parent, config_path=args.config, split_manifest=manifest, phase="flow", stage=stage, seed=seed, input_paths=[args.encoder, args.flow, args.gate]),
        "inference_input": "eeg_only",
        "reference_audio_used_for_synthesis": False,
        "split": args.split,
        "test_accessed": args.split == "subject_test",
        "files": saved,
    })
    print(str(output))


if __name__ == "__main__":
    main()
