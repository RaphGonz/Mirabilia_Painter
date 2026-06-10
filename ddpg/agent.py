import copy
import torch
import torch.nn as nn
from models.actor import Actor
from models.critic import Critic
from config import TAU, ACTOR_LR, CRITIC_LR, GAMMA, GRAD_CLIP_CRITIC


@torch.no_grad()
def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    """
    In-place tau-weighted blend: theta_target <- tau*theta + (1-tau)*theta_target.

    Decorated with @torch.no_grad() — gradients must never flow into target networks.
    Uses in-place ops (mul_ / add_) to avoid allocating new tensors.

    Args:
        target: Target network (frozen copy, updated by soft blend).
        source: Live network (params are read but not modified).
        tau:    Blend rate in (0, 1]. TAU=0.005 per paper Table 1.
    """
    for p_targ, p in zip(target.parameters(), source.parameters()):
        p_targ.data.mul_(1.0 - tau)
        p_targ.data.add_(tau * p.data)


class DDPGAgent:
    """
    DDPG agent scaffold for the painting task.

    Phase 3 scope: target network lifecycle (deepcopy, eval(), frozen params)
    and soft_update. The full actor/critic gradient update (update_step) is
    Phase 4 scope — it requires the frozen SoftRasterizer for actor loss
    computation (differentiable compositing of stroke outputs onto the canvas).

    Attributes:
        actor (Actor):           Live actor network (policy).
        critic (Critic):         Live critic network (value estimator V(s')).
        actor_target (Actor):    Deepcopy of actor, permanently eval() + frozen.
        critic_target (Critic):  Deepcopy of critic, permanently eval() + frozen.
        actor_opt:               Adam optimizer for actor (lr=ACTOR_LR).
        critic_opt:              Adam optimizer for critic (lr=CRITIC_LR).

    deepcopy safety: critic_target construction is safe because models/critic.py
    uses torch.nn.utils.parametrizations.weight_norm (modern API), NOT the
    deprecated torch.nn.utils.weight_norm which raises RuntimeError on deepcopy.
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

        # Live networks — trained during update_step (Phase 4)
        self.actor = Actor().to(device)
        self.critic = Critic().to(device)

        # Target networks — deepcopy of live nets, never trained directly.
        # deepcopy-safe: critic uses parametrizations.weight_norm (Plan 03-03).
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Permanently freeze both target networks.
        # Double-freeze pattern (mirrors pretrain_renderer.py::load_frozen_renderer):
        #   eval()               — disables BN/dropout training-mode behavior
        #   requires_grad_(False) — prevents any gradient accumulation into targets
        self.actor_target.eval()
        self.critic_target.eval()
        for p in self.actor_target.parameters():
            p.requires_grad_(False)
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # Adam optimizers for live networks (LR from paper Table 1, CONTEXT.md D-18)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=ACTOR_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=CRITIC_LR)

    def update_step(self, batch) -> None:
        """
        PHASE 4 PLACEHOLDER — raises NotImplementedError.

        This method documents the intended Phase 4 update sequence but does NOT
        implement it — the SoftRasterizer (frozen differentiable renderer) is
        required for actor loss computation and is wired in Phase 4.

        Intended Phase 4 sequence:
          obs, act, rew, next_obs, done = batch

          # --- Critic (Bellman target) ---
          with torch.no_grad():
              v_next = critic_target(next_obs)           # (B, 1) stable V(s')
              y = rew.unsqueeze(1) + GAMMA * v_next * (~done.unsqueeze(1))
          v_pred = critic(obs)                          # (B, 1) � V(s), NOT next_obs
          critic_loss = F.mse_loss(v_pred, y)
          critic_opt.zero_grad()
          critic_loss.backward()
          torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP_CRITIC)
          critic_opt.step()

          # --- Actor (policy gradient through frozen SoftRasterizer R) ---
          # actor_loss = -critic(render_next_state(actor(obs))).mean()
          # Where render_next_state applies R(actor(obs)) to canvas to produce s_{t+1}
          # actor_opt.zero_grad()
          # actor_loss.backward()
          # actor_opt.step()

          # --- Soft updates (after every critic step, per D-11) ---
          soft_update(actor_target,  actor,  TAU)
          soft_update(critic_target, critic, TAU)
        """
        raise NotImplementedError(
            "update_step() is a Phase 4 placeholder. "
            "Full critic Bellman update + actor policy gradient through the frozen "
            "SoftRasterizer (differentiable renderer) is Phase 4 scope. "
            "See the docstring for the intended implementation sequence."
        )
