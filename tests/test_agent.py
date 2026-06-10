import torch
import pytest
import copy
from ddpg.agent import DDPGAgent, soft_update
from config import TAU


def test_targets_are_deepcopies():
    """Target networks are distinct objects from live nets but equal at init."""
    agent = DDPGAgent(device=torch.device('cpu'))
    # Identity check — different objects
    assert agent.actor_target is not agent.actor
    assert agent.critic_target is not agent.critic
    # Values are equal at init (deepcopy preserves weights)
    p_actor = next(agent.actor.parameters()).data
    p_actor_t = next(agent.actor_target.parameters()).data
    assert torch.allclose(p_actor, p_actor_t), "actor_target params must equal actor at init"
    p_critic = next(agent.critic.parameters()).data
    p_critic_t = next(agent.critic_target.parameters()).data
    assert torch.allclose(p_critic, p_critic_t), "critic_target params must equal critic at init"


def test_target_eval_mode():
    """Both target networks are permanently in eval() mode."""
    agent = DDPGAgent(device=torch.device('cpu'))
    assert not agent.actor_target.training, "actor_target must be in eval() mode"
    assert not agent.critic_target.training, "critic_target must be in eval() mode"


def test_targets_frozen():
    """All parameters of both target networks have requires_grad == False."""
    agent = DDPGAgent(device=torch.device('cpu'))
    for name, p in agent.actor_target.named_parameters():
        assert not p.requires_grad, f"actor_target param {name} must have requires_grad=False"
    for name, p in agent.critic_target.named_parameters():
        assert not p.requires_grad, f"critic_target param {name} must have requires_grad=False"


def test_soft_update():
    """soft_update with TAU=0.005 produces (1-TAU)*target_old + TAU*source."""
    agent = DDPGAgent(device=torch.device('cpu'))
    # Clone the first target param before update
    p_before = next(agent.critic_target.parameters()).data.clone()
    # Read the corresponding source param
    p_source = next(agent.critic.parameters()).data
    # Apply soft update
    soft_update(agent.critic_target, agent.critic, TAU)
    # Read target param after
    p_after = next(agent.critic_target.parameters()).data
    # Verify: p_after == (1 - TAU) * p_before + TAU * p_source
    expected = (1 - TAU) * p_before + TAU * p_source
    assert torch.allclose(p_after, expected, atol=1e-6), (
        f"soft_update result mismatch. max diff: {(p_after - expected).abs().max().item()}"
    )


def test_critic_deepcopy_safe():
    """Constructing DDPGAgent must not raise — proves parametrizations.weight_norm works with deepcopy."""
    # This test would fail with deprecated torch.nn.utils.weight_norm
    try:
        agent = DDPGAgent(device=torch.device('cpu'))
    except RuntimeError as e:
        pytest.fail(f"DDPGAgent construction raised RuntimeError (deepcopy + WN bug): {e}")
    # Extra: verify critic_target is indeed a deep copy (not a reference)
    assert agent.critic_target is not agent.critic


def test_update_step_not_implemented():
    """update_step() must raise NotImplementedError — full update is Phase 4 scope."""
    agent = DDPGAgent(device=torch.device('cpu'))
    with pytest.raises(NotImplementedError):
        agent.update_step(None)


def test_agent_gpu():
    """Agent moves actor/critic to CUDA device when available."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    device = torch.device("cuda")
    agent = DDPGAgent(device=device)
    assert next(agent.actor.parameters()).device.type == "cuda", "actor must be on CUDA"
    assert next(agent.critic.parameters()).device.type == "cuda", "critic must be on CUDA"
