import unittest
import torch
from src.alignment import dpo_loss


class LossTests(unittest.TestCase):
    def test_winner_fit_improves_objective(self):
        ref=torch.tensor([1.,1.])
        self.assertLess(dpo_loss(torch.tensor([.8,1.]),ref,0,2.),dpo_loss(ref,ref,0,2.))
        self.assertGreater(dpo_loss(torch.tensor([.8,1.]),ref,1,2.),dpo_loss(ref,ref,1,2.))
