import torch

from torch_geometric.nn.models.ncn import (
    NCNLinkPredictor,
    common_neighbor_pool,
)
from torch_geometric.utils import subgraph


def _ones_mlp(mlp) -> None:
    for lin in mlp.lins:
        torch.nn.init.constant_(lin.weight, 1.0)
        if lin.bias is not None:
            torch.nn.init.zeros_(lin.bias)


def test_common_neighbor_pool():
    x = torch.tensor([
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 2.0],
        [3.0, 4.0],
        [9.0, 9.0],
    ])
    # One-direction edges, a self-loop, and the direct pair (0, 1).
    edge_index = torch.tensor([
        [0, 1, 0, 0],
        [2, 2, 0, 1],
    ])
    pairs = torch.tensor([
        [0, 0],
        [1, 4],
    ])
    out = common_neighbor_pool(x, edge_index, pairs)
    assert torch.equal(out[0], x[2])
    assert torch.equal(out[1], torch.zeros(2))

    two = torch.tensor([
        [0, 0, 1, 1],
        [2, 3, 2, 3],
    ])
    summed = common_neighbor_pool(x, two, pairs[:, :1])
    assert torch.equal(summed[0], x[2] + x[3])


def test_direct_edge_is_not_a_common_neighbor():
    x = torch.arange(4, dtype=torch.float).view(4, 1)
    edge_index = torch.tensor([
        [0, 1],
        [1, 0],
    ])
    pairs = torch.tensor([[0], [1]])
    out = common_neighbor_pool(x, edge_index, pairs)
    assert torch.equal(out, torch.zeros(1, 1))


def test_ncn_distinguishes_common_neighbor_features():
    # Same endpoint features and the same common-neighbor count.
    # Figure 3: the common-neighbor features differ.
    x = torch.tensor([
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [1.0],
        [3.0],
    ])
    edge_index = torch.tensor([
        [0, 1, 3, 4],
        [2, 2, 5, 5],
    ])
    pairs = torch.tensor([
        [0, 3],
        [1, 4],
    ])
    model = NCNLinkPredictor(1, 1, dropout=0.0)
    _ones_mlp(model.cn_mlp)
    _ones_mlp(model.pair_mlp)
    _ones_mlp(model.out_mlp)
    model.beta.data.fill_(1)

    scores = model(x, edge_index, pairs)
    assert scores.size() == (2, )
    assert not torch.allclose(scores[0], scores[1])


def test_ncn_shapes_reset_and_empty_pairs():
    model = NCNLinkPredictor(8, 16)
    assert str(model) == 'NCNLinkPredictor(8, 16, out_channels=1)'

    x = torch.randn(5, 8)
    edge_index = torch.tensor([
        [0, 1, 2],
        [1, 2, 3],
    ])
    pairs = torch.tensor([
        [0, 1],
        [1, 2],
    ])
    out = model(x, edge_index, pairs)
    assert out.size() == (2, )

    wide = NCNLinkPredictor(8, 16, out_channels=3, dropout=0.0)
    wide_out = wide(x, edge_index, pairs)
    assert wide_out.size() == (2, 3)

    empty_pairs = torch.empty(2, 0, dtype=torch.long)
    empty = model(x, edge_index, empty_pairs)
    assert empty.size() == (0, )

    model.beta.data.fill_(4)
    model.reset_parameters()
    assert model.beta.item() == 1


def test_self_pair_sums_neighbors():
    x = torch.tensor([[1.0], [10.0], [100.0]])
    edge_index = torch.tensor([
        [0, 1],
        [1, 0],
    ])
    out = common_neighbor_pool(x, edge_index, torch.tensor([[0], [0]]))
    assert torch.equal(out, x[1].view(1, -1))


def _local_subgraph(subset, edge_index, pairs, num_nodes):
    sub_edge_index, _ = subgraph(
        subset,
        edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes,
    )
    mapper = subset.new_full((num_nodes, ), -1)
    mapper[subset] = torch.arange(subset.numel())
    return sub_edge_index, mapper[pairs]


def test_ncn_local_subgraph():
    # Pair (0, 1) has common neighbor 2. Node 4 lies outside the subgraph.
    num_nodes = 5
    x = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [4.0, 5.0],
        [0.0, 0.0],
        [9.0, 9.0],
    ])
    edge_index = torch.tensor([
        [0, 2, 0, 4],
        [2, 1, 3, 0],
    ])
    pairs = torch.tensor([[0], [1]])
    full = common_neighbor_pool(x, edge_index, pairs)
    assert torch.equal(full, x[2].view(1, -1))

    subset = torch.tensor([0, 1, 2, 3])
    sub_edge_index, local_pairs = _local_subgraph(
        subset, edge_index, pairs, num_nodes)
    local = common_neighbor_pool(x[subset], sub_edge_index, local_pairs)
    assert torch.equal(local, full)
    assert int(local_pairs.max()) < subset.numel()

    model = NCNLinkPredictor(2, 4, dropout=0.0)
    local_x = x[subset].detach().requires_grad_(True)
    logits = model(local_x, sub_edge_index, local_pairs)
    assert logits.size(0) == local_pairs.size(1)
    logits.sum().backward()
    assert local_x.grad is not None

    # A sampled subgraph that missed the common neighbor has a zero pool.
    ends = torch.tensor([0, 1])
    budget_edge_index, budget_pairs = _local_subgraph(
        ends, edge_index, pairs, num_nodes)
    budget = common_neighbor_pool(x[ends], budget_edge_index, budget_pairs)
    assert torch.equal(budget, torch.zeros(1, 2))
