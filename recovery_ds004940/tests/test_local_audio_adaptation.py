import argparse
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'app'))
sys.path.insert(0, str(ROOT / 'tests'))
import adapt_local_audio as adapt


class LocalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_spectral_loss_is_differentiable_and_exact_match_is_zero(self):
        wave = torch.randn(1, 2048) * .1
        prediction = (wave + torch.randn_like(wave) * .01).requires_grad_()
        loss = adapt.spectral_loss(prediction, wave)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.)
        self.assertAlmostEqual(float(adapt.spectral_loss(wave, wave)), 0., places=6)

    def test_real_tiny_hubert_masked_loss_reaches_consumed_layers(self):
        from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
        model = HubertModel(HubertConfig(hidden_size=12, num_hidden_layers=9, num_attention_heads=3,
            intermediate_size=24, conv_dim=(12,12,12), conv_stride=(5,4,4), conv_kernel=(10,4,4),
            num_conv_pos_embedding_groups=3, num_conv_pos_embeddings=8, mask_time_prob=.05, layerdrop=0.))
        processor = Wav2Vec2FeatureExtractor()
        wave = torch.randn(1, 4000)
        with torch.no_grad():
            clean = model(processor(wave.numpy(), sampling_rate=16000, return_tensors='pt').input_values,
                          output_hidden_states=True).hidden_states[9]
        loss = adapt.teacher_loss(model, processor, {'wave':wave, 'teacher':clean}, 31)
        loss.backward()
        self.assertGreater(float(model.encoder.layers[8].feed_forward.output_dense.weight.grad.abs().sum()), 0.)

    def test_local_decoder_can_train_without_external_corpus(self):
        import aligned_speech as runner
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            from test_aligned_speech import AlignedTests
            AlignedTests().fixture(base)
            cfg = {'artifact_root':str(base), 'decoder':{'hidden':12,'layers':2},
                   'audio_initialization':'project_train_only','minimum_epochs':1,'patience':1}
            args = argparse.Namespace(manifest=None,cache=None,initialize=None,stage='adapt',seed=31,
                lr=.001,batch_size=2,epochs=1,max_steps=0,checkpoint_every=1,device='cpu',output=str(base/'local'))
            runner.train_audio(args,cfg)
            result = runner.load_payload(base/'local/best_checkpoint.pt')
            self.assertEqual(result['stage'],'adapt')
            self.assertIsNone(result['signature']['initialize'])


    def test_vocoder_training_exports_and_resumes_on_train_fold_only(self):
        import h5py
        import numpy as np
        import yaml
        from transformers import SpeechT5HifiGan, SpeechT5HifiGanConfig
        from test_aligned_speech import AlignedTests
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); _, cache, _ = AlignedTests().fixture(base)
            with h5py.File(cache, "a") as h5:
                for group in h5["targets"].values():
                    del group["mel"]
                    group.create_dataset("mel", data=np.zeros((80, 251), dtype="float32"))
                    group["wave"][:] = np.random.default_rng(31).normal(0, .01, 64000)
            weights = base / "base"
            model = SpeechT5HifiGan(SpeechT5HifiGanConfig(upsample_initial_channel=32,
                    upsample_rates=[4,4,4,4], upsample_kernel_sizes=[8,8,8,8],
                    resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1,3,5]]))
            model.save_pretrained(weights)
            cfg = base / "config.yaml"
            cfg.write_text(yaml.safe_dump(dict(schema_version="aligned_speech_v1", artifact_root=str(base))))
            args = argparse.Namespace(config=str(cfg),device="cpu",base=str(weights),output=str(base/"adapt"),
                    cache=str(cache),kind="hifigan",epochs=1,lr=1e-5,batch_size=1)
            adapt.train(args)
            import json
            report = json.loads((base / "adapt/adaptation_report.json").read_text())
            self.assertEqual(report["train_unique_audio"], 2)
            self.assertEqual(report["validation_unique_audio"], 2)
            self.assertFalse(report["test_used"])
            selected = adapt.tree_hash(base / "adapt/best")
            adapt.train(args)
            self.assertEqual(selected, adapt.tree_hash(base / "adapt/best"))
            restored = SpeechT5HifiGan.from_pretrained(base / "adapt/best", local_files_only=True)
            self.assertFalse(torch.equal(model.conv_post.weight, restored.conv_post.weight))


    def test_hubert_training_saves_reloadable_teacher(self):
        import h5py
        import numpy as np
        import yaml
        from transformers import HubertConfig, HubertModel, Wav2Vec2FeatureExtractor
        from test_aligned_speech import AlignedTests
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); _, cache, _ = AlignedTests().fixture(base)
            model = HubertModel(HubertConfig(hidden_size=12,num_hidden_layers=9,num_attention_heads=3,
                intermediate_size=24,conv_dim=(12,)*7,num_conv_pos_embedding_groups=3,
                num_conv_pos_embeddings=8,mask_time_prob=.05,layerdrop=0.))
            weights=base/"base"; model.save_pretrained(weights)
            processor=Wav2Vec2FeatureExtractor(); processor.save_pretrained(weights)
            wave=np.random.default_rng(31).normal(0,.01,(1,64000)).astype("float32")
            model.eval()
            with torch.no_grad():
                clean=model(processor(wave,sampling_rate=16000,return_tensors="pt").input_values,
                            output_hidden_states=True).hidden_states[9][0].numpy()
            with h5py.File(cache,"a") as h5:
                h5.attrs["teacher_sha256"]=adapt.tree_hash(weights)
                for group in h5["targets"].values():
                    del group["teacher"]; group.create_dataset("teacher",data=clean)
                    group["wave"][:]=wave[0]
            cfg=base/"config.yaml"
            cfg.write_text(yaml.safe_dump(dict(schema_version="aligned_speech_v1",artifact_root=str(base))))
            args=argparse.Namespace(config=str(cfg),device="cpu",base=str(weights),output=str(base/"adapt"),
                cache=str(cache),kind="hubert",epochs=1,lr=1e-5,batch_size=1)
            adapt.train(args)
            restored=HubertModel.from_pretrained(base/"adapt/best",local_files_only=True)
            self.assertTrue(hasattr(restored,"masked_spec_embed"))
            self.assertFalse(torch.equal(model.encoder.layers[8].feed_forward.output_dense.weight,
                                         restored.encoder.layers[8].feed_forward.output_dense.weight))

if __name__ == '__main__': unittest.main()
