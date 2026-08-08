"""Prove the from-scratch study modules match their library equivalents.

This is the whole point of writing them by hand. If these pass, then when
training misbehaves later you already know the optimizer, schedule, and loss
are not the cause.

Run: python -m pytest test_study_modules.py -v
"""
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

import optim as my_optim
import schedule as my_schedule
import loss as my_loss
import metrics as my_metrics


def _tiny_problem(seed: int = 0):
    """A fixed quadratic. Same start point, same gradients, every run."""
    torch.manual_seed(seed)
    target = torch.randn(8, 4)
    param = torch.randn(8, 4, requires_grad=True)
    return param, target


def _run(optimizer_factory, steps: int = 25, seed: int = 0):
    """Optimize the tiny problem and return the parameter trajectory."""
    param, target = _tiny_problem(seed)
    opt = optimizer_factory([param])
    history = []
    for _ in range(steps):
        loss = ((param - target) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        history.append(param.detach().clone())
    return history


class TestAdam:
    def test_matches_torch_adam(self):
        mine = _run(lambda p: my_optim.Adam(p, lr=1e-2))
        theirs = _run(lambda p: torch.optim.Adam(p, lr=1e-2))
        for step, (a, b) in enumerate(zip(mine, theirs)):
            assert torch.allclose(a, b, atol=1e-6), f"diverged at step {step}"

    def test_matches_torch_adam_with_weight_decay(self):
        mine = _run(lambda p: my_optim.Adam(p, lr=1e-2, weight_decay=0.1))
        theirs = _run(lambda p: torch.optim.Adam(p, lr=1e-2, weight_decay=0.1))
        for step, (a, b) in enumerate(zip(mine, theirs)):
            assert torch.allclose(a, b, atol=1e-6), f"diverged at step {step}"

    def test_matches_torch_with_paper_betas(self):
        betas = (0.9, 0.98)
        mine = _run(lambda p: my_optim.Adam(p, lr=1e-3, betas=betas, eps=1e-9))
        theirs = _run(
            lambda p: torch.optim.Adam(p, lr=1e-3, betas=betas, eps=1e-9)
        )
        for step, (a, b) in enumerate(zip(mine, theirs)):
            assert torch.allclose(a, b, atol=1e-6), f"diverged at step {step}"

    def test_bias_correction_matters_on_first_step(self):
        """Without bias correction the first step would be ~10x too small."""
        param, _ = _tiny_problem()
        start = param.detach().clone()
        opt = my_optim.Adam([param], lr=1e-2)
        ((param - 100.0) ** 2).mean().backward()
        opt.step()
        moved = (param.detach() - start).abs().mean()
        # with correction the first step is almost exactly lr
        assert math.isclose(moved.item(), 1e-2, rel_tol=1e-3)


class TestAdamW:
    def test_matches_torch_adamw(self):
        mine = _run(lambda p: my_optim.AdamW(p, lr=1e-2, weight_decay=0.1))
        theirs = _run(lambda p: torch.optim.AdamW(p, lr=1e-2, weight_decay=0.1))
        for step, (a, b) in enumerate(zip(mine, theirs)):
            assert torch.allclose(a, b, atol=1e-6), f"diverged at step {step}"

    def test_differs_from_adam_when_decay_is_on(self):
        """Coupled and decoupled decay are genuinely different algorithms."""
        adam = _run(lambda p: my_optim.Adam(p, lr=1e-2, weight_decay=0.1))
        adamw = _run(lambda p: my_optim.AdamW(p, lr=1e-2, weight_decay=0.1))
        assert not torch.allclose(adam[-1], adamw[-1], atol=1e-4)

    def test_identical_to_adam_without_decay(self):
        adam = _run(lambda p: my_optim.Adam(p, lr=1e-2, weight_decay=0.0))
        adamw = _run(lambda p: my_optim.AdamW(p, lr=1e-2, weight_decay=0.0))
        assert torch.allclose(adam[-1], adamw[-1], atol=1e-7)


class TestParamGroups:
    def test_splits_matrices_from_biases_and_norms(self):
        model = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
        groups = my_optim.param_groups_with_decay(model, weight_decay=0.1)
        assert groups[0]["weight_decay"] == 0.1
        assert groups[1]["weight_decay"] == 0.0
        # Linear.weight is the only 2D parameter
        assert all(p.dim() >= 2 for p in groups[0]["params"])
        # Linear.bias, LayerNorm.weight, LayerNorm.bias
        assert len(groups[1]["params"]) == 3


class TestNoamSchedule:
    def test_matches_torch_lambdalr(self):
        d_model, warmup = 64, 100

        torch_opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        torch_sched = torch.optim.lr_scheduler.LambdaLR(
            torch_opt, my_schedule.noam_lambda(d_model, warmup)
        )
        my_opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=1.0)
        my_sched = my_schedule.NoamScheduler(my_opt, d_model, warmup)

        for _ in range(250):
            assert math.isclose(
                torch_opt.param_groups[0]["lr"],
                my_opt.param_groups[0]["lr"],
                rel_tol=1e-9,
            )
            torch_sched.step()
            my_sched.step()

    def test_peaks_at_warmup(self):
        d_model, warmup = 512, 4000
        rates = [my_schedule.noam_lr(s, d_model, warmup) for s in range(1, 8001)]
        peak_step = rates.index(max(rates)) + 1
        # the two branches cross at warmup
        assert abs(peak_step - warmup) <= 1

    def test_rises_then_falls(self):
        d_model, warmup = 512, 400
        before = my_schedule.noam_lr(200, d_model, warmup)
        at = my_schedule.noam_lr(400, d_model, warmup)
        after = my_schedule.noam_lr(2000, d_model, warmup)
        assert before < at
        assert after < at

    def test_wider_model_gets_smaller_rate(self):
        narrow = my_schedule.noam_lr(1000, 128, 400)
        wide = my_schedule.noam_lr(1000, 512, 400)
        assert wide < narrow

    def test_rejects_zero_warmup(self):
        with pytest.raises(ValueError):
            my_schedule.noam_lambda(64, 0)


class TestCrossEntropy:
    def test_matches_torch_plain(self):
        torch.manual_seed(0)
        logits = torch.randn(20, 11)
        targets = torch.randint(0, 11, (20,))
        mine = my_loss.cross_entropy(logits, targets)
        theirs = F.cross_entropy(logits, targets)
        assert torch.allclose(mine, theirs, atol=1e-6)

    def test_matches_torch_with_ignore_index(self):
        torch.manual_seed(1)
        logits = torch.randn(20, 11)
        targets = torch.randint(0, 11, (20,))
        targets[::3] = 0  # make some padding
        mine = my_loss.cross_entropy(logits, targets, ignore_index=0)
        theirs = F.cross_entropy(logits, targets, ignore_index=0)
        assert torch.allclose(mine, theirs, atol=1e-6)

    def test_matches_torch_with_label_smoothing(self):
        torch.manual_seed(2)
        logits = torch.randn(20, 11)
        targets = torch.randint(0, 11, (20,))
        mine = my_loss.cross_entropy(logits, targets, label_smoothing=0.1)
        theirs = F.cross_entropy(logits, targets, label_smoothing=0.1)
        assert torch.allclose(mine, theirs, atol=1e-6)

    def test_matches_torch_with_both(self):
        torch.manual_seed(3)
        logits = torch.randn(40, 17)
        targets = torch.randint(0, 17, (40,))
        targets[::4] = 0
        mine = my_loss.cross_entropy(
            logits, targets, ignore_index=0, label_smoothing=0.1
        )
        theirs = F.cross_entropy(
            logits, targets, ignore_index=0, label_smoothing=0.1
        )
        assert torch.allclose(mine, theirs, atol=1e-6)

    def test_sequence_form_matches_torch(self):
        torch.manual_seed(4)
        logits = torch.randn(3, 7, 13)
        targets = torch.randint(0, 13, (3, 7))
        targets[:, -2:] = 0
        mine = my_loss.sequence_cross_entropy(
            logits, targets, ignore_index=0, label_smoothing=0.1
        )
        theirs = F.cross_entropy(
            logits.reshape(-1, 13),
            targets.reshape(-1),
            ignore_index=0,
            label_smoothing=0.1,
        )
        assert torch.allclose(mine, theirs, atol=1e-6)

    def test_padding_does_not_change_loss(self):
        """Appending pure padding must leave the loss untouched."""
        torch.manual_seed(5)
        logits = torch.randn(10, 9)
        targets = torch.randint(1, 9, (10,))
        base = my_loss.cross_entropy(logits, targets, ignore_index=0)

        padded_logits = torch.cat([logits, torch.randn(5, 9)])
        padded_targets = torch.cat([targets, torch.zeros(5, dtype=torch.long)])
        padded = my_loss.cross_entropy(padded_logits, padded_targets, ignore_index=0)
        assert torch.allclose(base, padded, atol=1e-6)

    def test_confident_correct_prediction_is_near_zero(self):
        logits = torch.tensor([[0.0, 100.0, 0.0]])
        targets = torch.tensor([1])
        assert my_loss.cross_entropy(logits, targets).item() < 1e-6

    def test_all_padding_returns_zero(self):
        logits = torch.randn(4, 5)
        targets = torch.zeros(4, dtype=torch.long)
        assert my_loss.cross_entropy(logits, targets, ignore_index=0).item() == 0.0


class TestMetrics:
    def test_perplexity_of_zero_loss_is_one(self):
        assert math.isclose(my_metrics.perplexity(0.0), 1.0)

    def test_perplexity_is_exp_of_loss(self):
        assert math.isclose(my_metrics.perplexity(2.0), math.exp(2.0))

    def test_uniform_model_perplexity_equals_vocab_size(self):
        """A model with no knowledge over V classes has perplexity V."""
        vocab = 50
        logits = torch.zeros(10, vocab)
        targets = torch.randint(0, vocab, (10,))
        loss = my_loss.cross_entropy(logits, targets)
        assert math.isclose(my_metrics.perplexity(loss), vocab, rel_tol=1e-5)

    def test_perplexity_overflow_is_inf(self):
        assert my_metrics.perplexity(1000.0) == float("inf")

    def test_bleu_perfect_match_is_100(self):
        sent = "the cat sat on the mat".split()
        assert math.isclose(my_metrics.corpus_bleu([sent], [sent]), 100.0)

    def test_bleu_no_overlap_is_zero(self):
        assert my_metrics.corpus_bleu(
            [["completely", "different", "words", "here"]],
            [["the", "cat", "sat", "down"]],
        ) == 0.0

    def test_bleu_clips_repeated_ngrams(self):
        """'the the the the' must not score full unigram precision."""
        cand = ["the"] * 4
        ref = "the cat sat down".split()
        assert my_metrics.corpus_bleu([cand], [ref]) < 30.0

    def test_bleu_penalizes_short_output(self):
        ref = "the cat sat on the mat today".split()
        full = my_metrics.corpus_bleu([ref], [ref])
        short = my_metrics.corpus_bleu([ref[:4]], [ref])
        assert short < full

    def test_bleu_matches_sacrebleu(self):
        sacrebleu = pytest.importorskip("sacrebleu")
        cands = [
            "the cat sat on the mat",
            "a quick brown fox jumps over the lazy dog",
        ]
        refs = [
            "the cat is on the mat",
            "the quick brown fox jumped over the lazy dog",
        ]
        mine = my_metrics.corpus_bleu(
            [c.split() for c in cands], [r.split() for r in refs]
        )
        theirs = sacrebleu.corpus_bleu(
            cands, [refs], tokenize="none", smooth_method="none"
        ).score
        assert abs(mine - theirs) < 0.1
