import pytest

torch = pytest.importorskip("torch")
from sql_agent.loss import group_advantages, grpo_loss


def test_group_std_sample_and_zero_variance():
    rewards = torch.tensor([0., 1., 3., 4., 9., 9., 9., 9.])
    actual = group_advantages(rewards, ["a"]*4 + ["b"]*4)
    expected = (rewards[:4] - 2) / torch.sqrt(torch.tensor(10/3))
    torch.testing.assert_close(actual[:4], expected)
    assert actual[4:].count_nonzero() == 0


def test_symmetric_clip_and_sequence_weighting():
    ratios = torch.tensor([[1.5, .5, 1.0], [1.5, .5, 1.0]])
    mask = torch.tensor([[1., 1., 0.], [1., 1., 1.]])
    logp = ratios.log().requires_grad_()
    # Sequence 1 positive advantage, sequence 2 negative advantage.
    loss, _ = grpo_loss(logp, torch.zeros_like(logp), logp.detach(),
                        torch.tensor([[1.], [-1.]]), mask, kl_coefficient=0)
    expected = (-(1.2 + .5)/2 + (1.5 + .8 + 1)/3)/2
    assert loss.item() == pytest.approx(expected)
    loss.backward()
    assert logp.grad[0, 0] == 0 and logp.grad[1, 1] == 0
    assert logp.grad[0, 1] != 0 and logp.grad[1, 0] != 0


def test_feedback_mask_applies_to_kl_and_policy_and_unequal_lengths():
    logp = torch.tensor([[-1., -10000., -2.], [-2., -1., -10000.]], requires_grad=True)
    mask = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
    old = logp.detach().clone()
    ref = torch.zeros_like(logp)
    loss, metrics = grpo_loss(logp, old, ref, torch.tensor([[1.], [-1.]]), mask)
    assert torch.isfinite(loss)
    expected_kl = ((torch.exp(torch.tensor(1.))-2) + (torch.exp(torch.tensor(2.))-3))/2
    torch.testing.assert_close(metrics["kl"], expected_kl)
    loss.backward()
    assert torch.all(logp.grad[~mask.bool()] == 0)
    assert torch.all(logp.grad[mask.bool()] != 0)
