# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training step's options (trainmulti.py):

- the lean step (--lean-clip) - the gradient clip and the AdamW step in a few kernel launches: the global gradient norm
  of one concatenation of the gradients (not a norm per tensor), the clip passed to fused AdamW as its grad_scale (the
  kernel divides the gradients by it - not a multiply per tensor), and one step counter shared by the parameters (not
  one per tensor, each incremented and, with found_inf, decremented on its own). On MPS, which has no foreach kernels,
  the default clip and step counters are a launch per tensor - about 700 a training step. The results match the
  default's up to rounding (the clip's multiply is the kernel's divide).
- the learning rate schedule (lr_factor, --schedule wsd --cooldown F) - the default linear warmup then cosine to 0.05,
  or warmup-stable-decay: constant after the warmup, then the last F of the steps cooled down to 0.05 by 1 - sqrt
"""

import math

import torch


# The learning rate schedule's floor (a factor of the peak)
LR_FLOOR = 0.05

# The lean clip's row length (its norm is the norm of the rows' norms)
NORM_ROW = 1024


# The learning rate factor of a step - the linear warmup, then the cosine to the floor (the default - its arithmetic
# unchanged), or warmup-stable-decay (wsd): constant, then the last cooldown share of the steps down to the floor by
# 1 - sqrt
def lr_factor(step, steps, warmup, schedule='cosine', cooldown=0.0):
    if step < warmup:
        return (step + 1) / warmup
    if schedule == 'cosine':
        progress = (step - warmup) / max(1, steps - warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    if schedule != 'wsd':
        raise ValueError(f'unknown schedule {schedule}')
    decay_start = steps - int(round(cooldown * steps))
    if step < decay_start:
        return 1.0
    progress = min(1.0, (step - decay_start) / max(1, steps - decay_start))
    return LR_FLOOR + (1 - LR_FLOOR) * (1 - math.sqrt(progress))


class LeanOptimizer(torch.optim.Optimizer):
    """The lean step - fused AdamW with one shared step counter per group and the clip as the kernel's grad_scale. step
    clips the gradients' global norm to max_norm and skips a step with a non-finite norm on the device (found_inf),
    with no read of the device; it returns the norm."""

    def __init__(self, groups, lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, max_norm=1.0):
        super().__init__(groups, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay})
        self.max_norm = max_norm

    # The parameters' gradients' global norm, from one concatenation - zero-padded to rows of NORM_ROW, the norm of
    # the rows' norms (one long float32 sum on the CPU is off by about 1e-4 at a million values)
    def grad_norm(self):
        grads = [parameter.grad.reshape(-1) for group in self.param_groups for parameter in group['params'] if parameter.grad is not None]
        count = sum(grad.numel() for grad in grads)
        padding = -count % NORM_ROW
        if padding:
            grads.append(grads[0].new_zeros(padding))
        return torch.linalg.vector_norm(torch.linalg.vector_norm(torch.cat(grads).view(-1, NORM_ROW), dim=1))

    @torch.no_grad()
    def step(self, closure=None):
        norm = self.grad_norm()
        # (clip_grad_norm_'s factor max_norm / (norm + 1e-6), clamped to 1 - as a divisor; found_inf skips the step)
        grad_scale = torch.clamp((norm + 1e-6) / self.max_norm, min=1.0)
        found_inf = (~torch.isfinite(norm)).float()
        for group in self.param_groups:
            self._adamw_step(group, grad_scale, found_inf)
        return norm

    # Fused AdamW over a group, its parameters sharing one step counter
    def _adamw_step(self, group, grad_scale, found_inf):
        parameters = [parameter for parameter in group['params'] if parameter.grad is not None]
        if not parameters:
            return
        counter = self._counter(group)
        for parameter in parameters:
            state = self.state[parameter]
            if 'exp_avg' not in state:
                state['exp_avg'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state['step'] = counter
        counter.add_(1)
        beta1, beta2 = group['betas']
        torch._fused_adamw_(  # pylint: disable=protected-access
            parameters, [parameter.grad for parameter in parameters], [self.state[parameter]['exp_avg'] for parameter in parameters],
            [self.state[parameter]['exp_avg_sq'] for parameter in parameters], [], [counter] * len(parameters), amsgrad=False,
            lr=group['lr'], beta1=beta1, beta2=beta2, weight_decay=group['weight_decay'], eps=group['eps'], maximize=False,
            grad_scale=grad_scale, found_inf=found_inf)
        counter.sub_(found_inf)

    # A group's shared step counter (the first parameter's state's - every parameter's state holds the same tensor)
    def _counter(self, group):
        state = self.state[group['params'][0]]
        if 'step' not in state:
            state['step'] = torch.zeros((), dtype=torch.float32, device=group['params'][0].device)
        return state['step']

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # (the loaded states each have their own step tensor - shared again, as the step makes them)
        for group in self.param_groups:
            if group['params']:
                first = self.state[group['params'][0]]
                for parameter in group['params'][1:]:
                    if 'step' in self.state[parameter] and 'step' in first:
                        self.state[parameter]['step'] = first['step']
