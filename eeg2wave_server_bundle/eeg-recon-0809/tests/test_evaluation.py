import importlib.util
import sys
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"app/src"))
MODULE=ROOT/"app/evaluate_joint.py"
spec=importlib.util.spec_from_file_location("evaluate_joint",MODULE)
evaluate=importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluate)


class TestEvaluationContracts(unittest.TestCase):
    def test_retrieval_reports_multi_positive_chance(self):
        prediction=torch.eye(4)
        target=torch.eye(4)
        result=evaluate.content_retrieval(prediction,target,["a","a","b","b"])
        self.assertEqual(result["r1"],1.0)
        self.assertEqual(result["unique_contents"],2)
        self.assertEqual(result["chance_r1"],0.5)

    def test_registered_dataset_mean_gate_does_not_require_degenerate_same_content(self):
        target=torch.stack([torch.zeros(2,3),torch.zeros(2,3),torch.ones(2,3),torch.ones(2,3)])
        prediction=target.clone()
        metrics=evaluate.template_metrics(prediction,target,["a","a","b","b"],
                                          ["wav-a","wav-a","wav-b","wav-b"],target.mean(0))
        self.assertFalse(metrics["same_content_template_gate_applicable"])
        name,passed=evaluate.registered_collapse_check(metrics,{"collapse_baseline":"dataset_mean","template_improvement_min":0.5})
        self.assertEqual(name,"registered_dataset_mean_collapse_baseline"); self.assertTrue(passed)

    def test_dataset_mean_uses_supplied_train_reference(self):
        target=torch.zeros(2,2,3); prediction=target.clone(); train_reference=torch.ones(2,3)
        metrics=evaluate.template_metrics(prediction,target,["a","b"],["wav-a","wav-b"],train_reference)
        self.assertEqual(metrics["dataset_mean_template_source"],"train_fold_mean")
        self.assertAlmostEqual(metrics["dataset_mean_template_mfcc_l1"],1.0)

    def test_same_content_is_leave_one_realization_out(self):
        target=torch.stack([torch.zeros(2,3),torch.ones(2,3),torch.full((2,3),2.0)])
        prediction=target.clone()
        metrics=evaluate.template_metrics(prediction,target,["a","a","a"],
                                          ["wav-0","wav-1","wav-2"],target.mean(0))
        self.assertTrue(metrics["same_content_template_gate_applicable"])
        self.assertEqual(metrics["same_content_template_pairs"],3)
        self.assertAlmostEqual(metrics["same_content_template_improvement"],1.0)

    def test_subject_probe_reports_chance(self):
        embeddings=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]])
        result=evaluate.leave_one_out_subject_probe(embeddings,["s1","s1","s2","s2"])
        self.assertEqual(result["accuracy"],1.0); self.assertEqual(result["chance"],0.5)

    def test_trial_diversity_separates_within_and_between_content(self):
        values=torch.tensor([[[0.,0.]],[[0.1,0.]],[[2.,0.]],[[2.1,0.]]])
        result=evaluate.pairwise_diversity(values,["a","a","b","b"])
        self.assertEqual(result["pairs"],6)
        self.assertGreater(result["between_content_distance"],result["within_content_distance"])
        self.assertGreater(result["between_over_within"],1.0)


if __name__=="__main__": unittest.main()
