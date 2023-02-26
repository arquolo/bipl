__all__ = [
    'Confusion', 'ConfusionGrad', 'accuracy', 'accuracy_balanced', 'iou',
    'kappa', 'kappa_quadratic_weighted'
]

import torch

from .base import Staged, to_index_sparse, to_prob_sparse

_EPS = torch.finfo(torch.float).eps


class Confusion(Staged):
    """Confusion Matrix. Returns 2d tensor"""
    def __call__(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        c, pred, true = to_index_sparse(pred, true)
        mat = torch.zeros(c, c, dtype=torch.long)
        mat = mat.index_put_((true, pred), torch.tensor(1), accumulate=True)
        return mat.double() / mat.sum()

    def collect(self, mat: torch.Tensor) -> dict[str, torch.Tensor]:
        c = mat.shape[0]
        return {f'cm{c}': mat, **super().collect(mat)}


class ConfusionGrad(Confusion):
    """Confusion Matrix which can be used for loss functions"""
    def __call__(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        c, pred, true = to_prob_sparse(pred, true)
        mat = torch.zeros(c, c, dtype=pred.dtype).index_add(0, true, pred)
        return mat.double() / mat.sum()


def accuracy(mat: torch.Tensor) -> torch.Tensor:
    return mat.diag().sum() / mat.sum().clamp(_EPS)


def accuracy_balanced(mat: torch.Tensor) -> torch.Tensor:
    return (mat.diag() / mat.sum(1).clamp(_EPS)).mean()


def kappa(mat: torch.Tensor) -> torch.Tensor:
    expected = mat.sum(0) @ mat.sum(1)
    observed = mat.diag().sum()
    return 1 - (1 - observed) / (1 - expected).clamp(_EPS)


def kappa_quadratic_weighted(mat: torch.Tensor) -> torch.Tensor:
    y, x = map(torch.arange, mat.shape)

    weights = (y[:, None] - x[None, :]).double() ** 2
    weights /= weights.max()

    expected = mat.sum(0) @ weights @ mat.sum(1)
    observed = mat.view(-1) @ weights.view(-1)
    return 1 - observed / expected.clamp(_EPS)


def iou(mat: torch.Tensor) -> torch.Tensor:
    return mat.diag() / (mat.sum(0) + mat.sum(1) - mat.diag()).clamp(_EPS)
