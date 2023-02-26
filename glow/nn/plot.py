__all__ = ['plot_model']

import functools
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack

import graphviz
import torch
from torch import nn
from torch.autograd import Function

from .. import mangle, si

# TODO: Still buggy, continue research/refactor


def id_(x) -> str:
    if hasattr(x, 'variable'):
        x = x.variable
    addr = x.storage().data_ptr() if isinstance(x, torch.Tensor) else id(x)
    return hex(addr)


def flatten(xs) -> Iterator[torch.Tensor]:
    if xs is None:
        return
    if isinstance(xs, torch.Tensor):
        yield xs
        return
    if isinstance(xs, Iterable):
        if isinstance(xs, Mapping):
            xs = xs.items()
        for x in xs:
            yield from flatten(x)
        return
    raise TypeError(f'Unsupported argument type: {type(xs)}')


def sized(var: torch.Tensor):
    if max(var.shape) == var.numel():
        return f'{tuple(var.shape)}'
    return f'{tuple(var.shape)}\n{si(var.numel())}'


class Builder:
    def __init__(self,
                 inputs: set[str],
                 params: dict[str, str],
                 nesting: bool = True,
                 variables: bool = True):
        self.inputs = inputs
        self.params = params
        self.nesting = nesting
        self.variables = variables

        self._mangle = mangle()
        self._memo: dict[str, str] = {}
        self._shapes: dict[Function, str] = {}
        root = graphviz.Digraph(
            name='root',
            graph_attr={
                'rankdir': 'LR',
                'newrank': 'true',
                'color': 'lightgrey',
            },
            edge_attr={
                'labelfloat': 'true',
            },
            node_attr={
                'shape': 'box',
                'style': 'filled',
                'fillcolor': 'lightgrey',
                'fontsize': '12',
                'height': '0.2',
                'ranksep': '0.1',
            },
        )
        self.stack = [root]

    def _add_op_node(self, grad_id: str, grad: Function):
        label = type(grad).__name__.replace('Backward', '')
        if grad in self._shapes:
            label = f'{label}\n=> {tuple(self._shapes[grad])}'
        self.stack[-1].node(grad_id, label)

    def _add_var_node(self, var_id: str, var: torch.Tensor):
        label_ = []
        if param_name := self.params.get(var_id):
            root = self.stack[-1]
            parts = param_name.split('.')
            label_.append(parts[-1] if self.nesting else '.'.join(parts[1:]))
        else:
            root = self.stack[0]  # unnamed, that's why external

        label = '\n'.join([*label_, var_id, sized(var)])
        color = 'yellow' if var_id in self.inputs else 'lightblue'
        root.node(var_id, label, fillcolor=color)

    def _traverse_saved(self, grad_id: str, *tensors):
        tensors = *(v for v in tensors if isinstance(v, torch.Tensor)),
        if not tensors:
            return
        s_ctx = self.stack[-1].subgraph()
        assert s_ctx is not None
        with s_ctx as s:
            s.attr(rank='same')
            for var in tensors:
                var_id = id_(var)
                if var_id not in self._memo:
                    label = f'{var_id}\n{sized(var)}'
                    s.node(var_id, label, fillcolor='orange')
                s.edge(var_id, grad_id)

    def _traverse(self, grad: Function, depth: int = 0):
        if grad is None or (grad_id := id_(grad)) in self._memo:
            return

        root = self.stack[-1]
        self._memo[grad_id] = head = root.name
        if hasattr(grad, 'variable'):
            # Has variable, so it's either Parameter or Variable
            self._add_var_node(grad_id, grad.variable)
            yield (depth - 1, None, grad)
            return

        # Doesn't have variable, so it's "operation"
        self._add_op_node(grad_id, grad)

        # TODO : add merging of tensors with same data
        if self.variables and hasattr(grad, 'saved_tensors'):
            self._traverse_saved(grad_id, *(grad.saved_tensors or ()))

        for grad_next, _ in getattr(grad, 'next_functions', ()):
            if grad_next is None:
                continue
            yield from self._traverse(grad_next, depth + 1)

            next_id = id_(grad_next)
            tail = self._memo.get(next_id)
            if tail is not None and head is not None and not (
                    head.startswith(tail) or tail.startswith(head)):
                yield (depth, grad_next, grad)  # leafs, yield for depth-check
                continue

            name = self.params.get(next_id)
            if self.nesting and name and name.rpartition('.')[0] == head:
                s_ctx = root.subgraph()
                assert s_ctx is not None
                with s_ctx as s:
                    s.attr(rank='same')
                    s.edge(next_id, grad_id)  # same module, same rank
            else:
                self.stack[0].edge(next_id, grad_id)

    def _mark(self, ts):
        edges = []
        for t in flatten(ts):
            if t.grad_fn is not None:
                self._shapes[t.grad_fn] = t.shape  # type: ignore[assignment]
                edges += self._traverse(t.grad_fn)
        if not edges:
            return

        max_depth = max(depth for depth, *_ in edges) + 1
        for depth, tail, head in edges:  # inter-module edges
            if tail is not None:
                minlen = f'{max_depth - depth}' if self.nesting else None
                self.stack[0].edge(id_(tail), id_(head), minlen=minlen)

    def forward_pre(self, name, module, xs):
        self._mark(xs)
        # -------- start node --------
        if not self.nesting:
            return
        scope = graphviz.Digraph(name=self._mangle(name))
        scope.attr(label=f'{name.split(".")[-1]}:{type(module).__name__}')
        self.stack.append(scope)

    def forward(self, module, _, ys):
        self._mark(ys)
        if not self.nesting:
            return
        cluster = self.stack.pop()
        cluster.name = f'cluster_{cluster.name}'
        self.stack[-1].subgraph(cluster)
        # -------- end node --------


def plot_model(model: nn.Module,
               *input_shapes: tuple[int, ...],
               device='cpu',
               nesting: bool = True,
               variables: bool = False):
    """Produces Graphviz representation of PyTorch autograd graph

    Blue nodes are the Variables that require grad, orange are Tensors
    saved for backward in torch.autograd.Function
    """
    inputs = [
        torch.zeros(1, *s, device=device, requires_grad=True)
        for s in input_shapes
    ]
    params = model.state_dict(prefix='root.', keep_vars=True)
    hk = Builder(
        {id_(var) for var in inputs},
        {id_(var): name for name, var in params.items()},
        nesting=nesting,
        variables=variables,
    )
    with ExitStack() as stack:
        for name, m in model.named_modules(prefix='root'):
            stack.callback(
                m.register_forward_pre_hook(
                    functools.partial(hk.forward_pre, name)).remove)
            stack.callback(m.register_forward_hook(hk.forward).remove)
        model(*inputs)

    dot = hk.stack.pop()
    assert not hk.stack

    dot.filename = getattr(model, 'name', type(model).__qualname__)
    dot.directory = 'graphs'
    dot.format = 'svg'

    size_min = 12
    scale_factor = .15
    size = max(size_min, len(dot.body) * scale_factor)

    dot.graph_attr.update(size=f'{size},{size}')
    dot.render(cleanup=True)
    return dot
