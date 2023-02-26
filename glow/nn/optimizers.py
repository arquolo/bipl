__all__ = ['AdamW', 'RAdam', 'SGDW']

import torch
from torch.optim import optimizer


class _OptimizerBase(optimizer.Optimizer):
    _step = 0

    def __getstate__(self):
        return {**super().__getstate__(), '_step': self._step}

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self._step += 1
        for group in self.param_groups:
            args = self._update_group(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Sparse grads are not supported')
                with torch.no_grad():
                    self._do_step(p, group, self.state[p], *args)

        return loss  # noqa: R504

    def _update_group(self, group) -> tuple:
        return ()

    def _do_step(self, p, group, state, *args):
        raise NotImplementedError


class SGDW(_OptimizerBase):
    """Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf
    """
    def __init__(self,
                 params,
                 lr=0.003,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False):
        assert lr >= 0.0
        assert momentum >= 0.0
        assert weight_decay >= 0.0
        defaults = {
            'lr': lr,
            'momentum': momentum,
            'dampening': dampening,
            'weight_decay': weight_decay,
            'nesterov': nesterov,
        }
        assert (not nesterov or (momentum > 0 and dampening == 0)
                ), 'Nesterov momentum requires a momentum and zero dampening'
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def _do_step(self, p, group, state):
        if group['weight_decay'] != 0:
            p.mul_(1 - group['lr'] * group['weight_decay'])

        momentum = group['momentum']
        grad = p.grad
        if momentum != 0:
            if state:
                grad = state['exp_avg']
                grad.mul_(momentum).add_(p.grad, alpha=1 - group['dampening'])
            else:
                grad = state['exp_avg'] = p.grad.clone().detach_()

            if group['nesterov']:
                grad = p.grad.add(grad, alpha=momentum)

        p.add_(grad, alpha=-group['lr'])


class AdamW(_OptimizerBase):
    r"""Implements AdamW algorithm.

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=1e-2,
                 amsgrad=False) -> None:
        assert 0.0 <= lr
        assert 0.0 <= eps
        for i, beta in enumerate(betas):
            assert 0.0 <= beta < 1.0, f'Invalid beta at index {i}: {betas}'
        defaults = {
            'lr': lr,
            'betas': betas,
            'eps': eps,
            'weight_decay': weight_decay,
            'amsgrad': amsgrad,
        }
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def _update_group(self, group) -> tuple:
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** self._step
        bias_correction2 = 1 - beta2 ** self._step
        return group['lr'] * (bias_correction2 ** 0.5) / bias_correction1,

    def _do_step(self, p, group, state, step_size):
        amsgrad = group['amsgrad']
        if not state:
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)
            if amsgrad:
                state['max_exp_avg_sq'] = torch.zeros_like(p)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

        if group['weight_decay'] != 0:
            p.mul_(1 - group['lr'] * group['weight_decay'])

        if amsgrad:
            max_exp_avg_sq = state['max_exp_avg_sq']
            torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = max_exp_avg_sq.sqrt().add_(group['eps'])
        else:
            denom = exp_avg_sq.sqrt().add_(group['eps'])

        p.addcdiv_(-step_size, exp_avg, denom)


class RAdam(_OptimizerBase):
    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 weight_decay=0.0,
                 decay_to_sgd=True) -> None:
        self._decay_to_sgd = decay_to_sgd
        defaults = {'lr': lr, 'betas': betas, 'weight_decay': weight_decay}
        super().__init__(params, defaults)

    def _update_group(self, group):
        beta1, beta2 = group['betas']
        bias_correction1 = 1 - beta1 ** self._step
        bias_correction2 = 1 - beta2 ** self._step

        beta2_t = beta2 ** self._step
        n_sma_max = 2 / (1 - beta2) - 1
        n_sma = n_sma_max - 2 * self._step * beta2_t / bias_correction2

        # more conservative since it's an approximated value
        # variance is not tractable
        if n_sma < 5:
            return False, (1 / bias_correction1)

        k = (n_sma - 4) * (n_sma - 2) / n_sma
        k_max = (n_sma_max - 4) * (n_sma_max - 2) / n_sma_max
        step_size = ((1 - beta2_t) * k / k_max) ** 0.5 / bias_correction1
        return True, step_size

    def _do_step(self, p, group, state, is_tractable, step_size):
        if not state:
            state['exp_avg'] = torch.zeros_like(p)
            state['exp_avg_sq'] = torch.zeros_like(p)

        exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        beta1, beta2 = group['betas']

        exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

        if group['weight_decay'] != 0:
            p.data.mul_(1 - group['weight_decay'] * group['lr'])

        if is_tractable:
            denom = exp_avg_sq.sqrt().add_(1e-8)
            p.addcdiv_(exp_avg, denom, value=-step_size * group['lr'])
        elif self._decay_to_sgd:
            p.add_(exp_avg, alpha=-step_size * group['lr'])
