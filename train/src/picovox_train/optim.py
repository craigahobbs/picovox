# Licensed under the MIT License
# https://github.com/craigahobbs/picovox/blob/main/LICENSE

"""
The training step's options (trainmulti.py and distill.py):

- the lean step (--lean-clip) - the gradient clip and the AdamW step in a few kernel launches: the global gradient norm
  of one concatenation of the gradients (not a norm per tensor), the clip passed to fused AdamW as its grad_scale (the
  kernel divides the gradients by it - not a multiply per tensor), and one step counter shared by the parameters (not
  one per tensor, each incremented and, with found_inf, decremented on its own). On MPS, which has no foreach kernels,
  the default clip and step counters are a launch per tensor - about 700 a teacher step, 600 a student step. The
  results match the default's up to rounding (the clip's multiply is the kernel's divide).
- Muon (the student's optimizer) - the hidden Linear matrices (the conv-FFN blocks, attention, FiLM, the condition and
  duration maps, the decoder's input) orthogonalized by Newton-Schulz, the rest AdamW (the tables, the head, the
  depthwise convolutions, the norms, and the biases); all the matrices at once, zero-padded to one [N, rows, columns]
  batch (the odd polynomial keeps the zero blocks zero) - about 150 launches a step, against torch.optim.Muon's loop
  per matrix. The update is scaled to AdamW's RMS (0.2 sqrt(max(rows, columns)) - match_rms_adamw), so the AdamW
  learning rate and weight decay carry over; Nesterov momentum, warmed up from 0.85 to 0.95 over the first 300 steps.
  It takes the lean step's clip and skips a non-finite step on the device, as fused AdamW does.
- the learning rate schedule (lr_factor, --schedule wsd --cooldown F) - the default linear warmup then cosine to 0.05,
  or warmup-stable-decay: constant after the warmup, then the last F of the steps cooled down to 0.05 by 1 - sqrt
"""

import math

import torch


# The learning rate schedule's floor (a factor of the peak)
LR_FLOOR = 0.05

# The lean clip's row length (its norm is the norm of the rows' norms)
NORM_ROW = 1024

# Muon's Newton-Schulz quintic coefficients and iterations (Keller Jordan's)
NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
NS_STEPS = 5

# Muon's momentum, warmed up from MUON_MOMENTUM_START over MUON_MOMENTUM_WARMUP steps
MUON_MOMENTUM = 0.95
MUON_MOMENTUM_START = 0.85
MUON_MOMENTUM_WARMUP = 300


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


# The hidden Linear matrices of a MultiModel (Muon's parameters) - every trainable 2-D Linear weight but the head, the
# duration output, and the position input (their inputs or outputs are a handful of features)
def hidden_matrices(model):
    excluded = {id(model.head.weight), id(model.duration_out.weight), id(model.position_in.weight)}
    matrices = []
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.weight.requires_grad and id(module.weight) not in excluded:
            matrices.append(module.weight)
    return matrices


# The optimizer's param groups for a model's trainable parameters - one AdamW group, or with muon the hidden matrices as
# a second group
def param_groups(model, muon=False):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not muon:
        return [{'params': trainable}]
    matrices = {id(parameter) for parameter in hidden_matrices(model)}
    return [{'params': [parameter for parameter in trainable if id(parameter) not in matrices]},
            {'params': [parameter for parameter in trainable if id(parameter) in matrices], 'muon': True}]


# A batch [N, R, C] orthogonalized by the Newton-Schulz quintic (each matrix's Frobenius norm first)
def newton_schulz(batch, steps=NS_STEPS):
    coefficient_a, coefficient_b, coefficient_c = NS_COEFFICIENTS
    batch = batch / (torch.linalg.matrix_norm(batch, keepdim=True) + 1e-7)
    for _ in range(steps):
        gram = torch.bmm(batch, batch.mT)
        polynomial = torch.baddbmm(gram, gram, gram, beta=coefficient_b, alpha=coefficient_c)
        batch = torch.baddbmm(batch, polynomial, batch, beta=coefficient_a)
    return batch


class LeanOptimizer(torch.optim.Optimizer):
    """The lean step (and Muon) - fused AdamW for the AdamW groups with one shared step counter per group and the clip
    as the kernel's grad_scale; Muon for the groups marked muon, all their matrices zero-padded into one batch. step
    clips the gradients' global norm to max_norm and skips a step with a non-finite norm on the device (found_inf),
    with no read of the device; it returns the norm."""

    def __init__(self, groups, lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=0.0, max_norm=1.0):
        super().__init__(groups, {'lr': lr, 'betas': betas, 'eps': eps, 'weight_decay': weight_decay, 'muon': False})
        self.max_norm = max_norm
        # (each muon group's padded layout - its matrices' sizes and orientation - made at its first step)
        self.layouts = {}

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
        for index, group in enumerate(self.param_groups):
            if group['muon']:
                self._muon_step(index, group, grad_scale, found_inf)
            else:
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

    # Muon over a group - its matrices (transposed to rows <= columns) zero-padded into one batch: the gradients
    # unclipped, Nesterov momentum, Newton-Schulz, the AdamW-matched scale, the decoupled weight decay
    def _muon_step(self, index, group, grad_scale, found_inf):
        parameters = group['params']
        if index not in self.layouts:
            sizes = [tuple(parameter.shape) for parameter in parameters]
            transposed = [rows > columns for rows, columns in sizes]
            oriented = [(columns, rows) if flip else (rows, columns) for (rows, columns), flip in zip(sizes, transposed)]
            shape = (len(parameters), max(rows for rows, _ in oriented), max(columns for _, columns in oriented))
            scale = torch.tensor([0.2 * math.sqrt(max(size)) for size in sizes], device=parameters[0].device)
            self.layouts[index] = {'transposed': transposed, 'oriented': oriented, 'shape': shape, 'scale': scale[:, None, None],
                                   'grads': torch.zeros(shape, device=parameters[0].device)}
        layout = self.layouts[index]
        state = self.state[parameters[0]]
        if 'momentum_buffer' not in state:
            state['momentum_buffer'] = torch.zeros(layout['shape'], device=parameters[0].device)
            state['muonStep'] = 0
        grads = layout['grads']
        for number, parameter in enumerate(parameters):
            rows, columns = layout['oriented'][number]
            grad = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
            grads[number, :rows, :columns].copy_(grad.mT if layout['transposed'][number] else grad)
        # (a non-finite step's gradients are zeroed and its updates scaled by zero - skipped, without a read)
        keep = 1 - found_inf
        grads.div_(grad_scale).nan_to_num_(0.0, 0.0, 0.0)
        momentum = MUON_MOMENTUM_START + (MUON_MOMENTUM - MUON_MOMENTUM_START) * min(1.0, state['muonStep'] / MUON_MOMENTUM_WARMUP)
        buffer = state['momentum_buffer']
        buffer.lerp_(grads, keep * (1 - momentum))
        update = newton_schulz(grads.lerp(buffer, momentum))
        update.mul_(layout['scale'] * (-group['lr'] * keep))
        decay = 1 - group['lr'] * group['weight_decay'] * keep
        for number, parameter in enumerate(parameters):
            rows, columns = layout['oriented'][number]
            block = update[number, :rows, :columns]
            parameter.mul_(decay).add_(block.mT if layout['transposed'][number] else block)
        state['muonStep'] += 1

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # (the loaded states each have their own step tensor - shared again, as the step makes them)
        for group in self.param_groups:
            if not group['muon'] and group['params']:
                first = self.state[group['params'][0]]
                for parameter in group['params'][1:]:
                    if 'step' in self.state[parameter] and 'step' in first:
                        self.state[parameter]['step'] = first['step']
