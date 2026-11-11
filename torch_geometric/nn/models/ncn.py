from typing import Optional

import torch
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.index import index2ptr
from torch_geometric.nn.models.mlp import MLP
from torch_geometric.utils import (
    coalesce,
    remove_self_loops,
    scatter,
    to_undirected,
)


def common_neighbor_pool(
    x: Tensor,
    edge_index: Tensor,
    edge_label_index: Tensor,
) -> Tensor:
    r"""Sums embeddings over the first-order common neighbors of each pair.

    :obj:`edge_index` may store each undirected edge in one direction. The
    pool symmetrizes it, drops self-loops, and ignores edge weights. Neither
    endpoint is a common neighbor of its pair. A pair :obj:`(i, i)` sums
    :math:`N(i)`.

    One index is materialized per neighbor of each source endpoint. Batch
    :obj:`edge_label_index` on a large graph.

    Args:
        x (torch.Tensor): Node embeddings
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        edge_index (torch.Tensor): Graph connectivity in COO format.
        edge_label_index (torch.Tensor): Node pairs to score, of shape
            :obj:`[2, num_pairs]`.

    Returns:
        torch.Tensor: Pooled embeddings of shape :obj:`[num_pairs, F]`.
        An empty intersection is a zero row.
    """
    num_nodes = x.size(0)
    channels = x.size(-1)
    num_pairs = edge_label_index.size(1)
    if num_nodes > 2**31:
        raise ValueError(f"Common-neighbor pooling supports at most 2**31 "
                         f"nodes (got {num_nodes})")

    out = x.new_zeros(num_pairs, channels)
    if num_pairs == 0 or edge_index.numel() == 0:
        return out

    edge_index = edge_index.to(x.device)
    edge_label_index = edge_label_index.to(x.device)
    undirected = to_undirected(edge_index, num_nodes=num_nodes)
    edge_index, _ = remove_self_loops(undirected)
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    if edge_index.numel() == 0:
        return out

    row, col = edge_index[0].long(), edge_index[1].long()
    rowptr = index2ptr(row, size=num_nodes)
    src = edge_label_index[0].long()
    dst = edge_label_index[1].long()

    node_deg = (rowptr[1:] - rowptr[:-1])[src]
    total = int(node_deg.sum())
    if total == 0:
        return out

    pair_idx = torch.arange(num_pairs, device=x.device)
    pair_idx = pair_idx.repeat_interleave(node_deg)
    starts = rowptr[src].repeat_interleave(node_deg)
    exclusive = torch.cumsum(node_deg, dim=0) - node_deg
    offsets = torch.arange(total, device=x.device)
    offsets = offsets - exclusive.repeat_interleave(node_deg)
    neighbors = col[starts + offsets]

    keys = torch.sort(row * num_nodes + col).values
    query = dst[pair_idx] * num_nodes + neighbors
    pos = torch.searchsorted(keys, query)
    in_range = pos < keys.numel()
    pos = pos.clamp(max=keys.numel() - 1)
    matched = in_range & (keys[pos] == query)
    if not bool(matched.any()):
        return out

    return scatter(
        x[neighbors[matched]],
        pair_idx[matched],
        dim=0,
        dim_size=num_pairs,
        reduce='sum',
    )


class NCNLinkPredictor(torch.nn.Module):
    r"""The Neural Common Neighbor (NCN) link predictor from the
    `"Neural Common Neighbor with Completion for Link Prediction"
    <https://arxiv.org/abs/2302.00890>`_ paper.

    Scores a pair :math:`(i, j)` from embeddings of one MPNN run:

    .. math::

        \mathrm{MLP}\bigl(\beta \cdot \mathrm{MLP}(
        \textstyle\sum_{u \in N(i) \cap N(j)} \mathbf{x}_u)
        + \mathrm{MLP}(\mathbf{x}_i \odot \mathbf{x}_j)\bigr).

    :obj:`beta` is a scalar parameter initialized to :obj:`1`. Each MLP has
    two layers. Equation 9 of the paper concatenates the sum and the product.
    The tables also train a node residual, layer norm, and a tail activation.

    Pass embeddings computed on the target-masked graph, for example from
    :class:`~torch_geometric.transforms.RandomLinkSplit` with
    :obj:`disjoint_train_ratio` greater than :obj:`0`, and pass that same
    :obj:`edge_index`. A neighbor-sampled :obj:`edge_index` includes only
    common neighbors that remain in the subgraph. An empty intersection
    leaves the product term. Keep :obj:`norm` at :obj:`None` so scores do
    not depend on batch statistics of the pair batch.

    Args:
        in_channels (int): Size of each input embedding.
        hidden_channels (int): Size of each hidden embedding.
        out_channels (int, optional): Size of each output.
            (default: :obj:`1`)
        dropout (float, optional): Dropout probability.
            (default: :obj:`0.0`)
        norm (str, optional): Normalization passed to each MLP.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 1,
        dropout: float = 0.0,
        norm: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

        self.cn_mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            plain_last=True,
        )
        self.pair_mlp = MLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            plain_last=True,
        )
        self.out_mlp = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=2,
            dropout=dropout,
            norm=norm,
            plain_last=True,
        )
        self.beta = Parameter(torch.ones(1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Resets all learnable parameters of the module."""
        self.cn_mlp.reset_parameters()
        self.pair_mlp.reset_parameters()
        self.out_mlp.reset_parameters()
        self.beta.data.fill_(1)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_label_index: Tensor,
    ) -> Tensor:
        r"""Scores each pair in :obj:`edge_label_index`.

        Args:
            x (torch.Tensor): Node embeddings from one MPNN run.
            edge_index (torch.Tensor): Target-masked graph connectivity.
            edge_label_index (torch.Tensor): Pairs to score.

        Returns:
            torch.Tensor: Logits of shape :obj:`[num_pairs]` when
            :obj:`out_channels` is :obj:`1`, otherwise
            :obj:`[num_pairs, out_channels]`.
        """
        pool = common_neighbor_pool(x, edge_index, edge_label_index)
        src, dst = edge_label_index[0], edge_label_index[1]
        pair = self.pair_mlp(x[src] * x[dst])
        structural = self.cn_mlp(pool)
        out = self.out_mlp(self.beta * structural + pair)
        if self.out_channels == 1:
            return out.view(-1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.hidden_channels}, out_channels={self.out_channels})')
