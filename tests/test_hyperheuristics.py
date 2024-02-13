import pytest
import torch

from torch_utils import get_batch_tensor_from_routes
from learning.hyperheuristics import count_violations


def test_no_routes():
    routes = [[]]
    routes = get_batch_tensor_from_routes(routes)
    demand = 5 * (1 - torch.eye(0, dtype=int))

    # no nodes!
    nv = count_violations(routes, 0, 0, torch.zeros((1, 0, 0), dtype=int),
                          demand)
    assert nv == 0

    # two nodes but no routes required.  Still results in some violations.
    sp_lens = 2 * (1 - torch.eye(2, dtype=int))
    demand = 5 * (1 - torch.eye(2, dtype=int))
    nv = count_violations(routes, 2, 0, sp_lens[None], demand)
    # 2 nodes uncovered, 1 node pair unlinked
    assert nv == 3

    # five nodes but no routes required.  Still results in some violations.
    sp_lens = 2 * (1 - torch.eye(5, dtype=int))
    demand = 5 * (1 - torch.eye(5, dtype=int))
    nv = count_violations(routes, 5, 0, sp_lens[None], demand)
    # 5 nodes uncovered, 10 node pairs unlinked
    assert nv == 15

    # two nodes and two routes required.
    sp_lens = 2 * (1 - torch.eye(2, dtype=int))
    demand = 5 * (1 - torch.eye(2, dtype=int))
    nv = count_violations(routes, 2, 2, sp_lens[None], demand)
    # 2 nodes uncovered, 1 node pair unlinked, 2 routes missing
    assert nv == 5


def test_coverage():
    routes = [[0, 3, 2]]
    routes = get_batch_tensor_from_routes(routes)

    # one node uncovered, so three node pairs unlinked
    sp_lens = 2 * (1 - torch.eye(4, dtype=int))[None]
    demand = 5 * (1 - torch.eye(4, dtype=int))
    nv = count_violations(routes, 4, 1, sp_lens, demand)
    assert nv == 4

    routes = [[0, 3, 1, 2]]
    routes = get_batch_tensor_from_routes(routes)

    # everything correct
    nv = count_violations(routes, 4, 1, sp_lens, demand)
    assert nv == 0

    # three nodes uncovered, 15 pairs unlinked
    sp_lens = 2 * (1 - torch.eye(7, dtype=int))[None]
    demand = 5 * (1 - torch.eye(7, dtype=int))
    nv = count_violations(routes, 7, 1, sp_lens, demand)
    assert nv == 3 + 15


def test_connected():
    routes = [[0, 1],
              [2, 3]]
    demand = 5 * (1 - torch.eye(4, dtype=int))
    routes = get_batch_tensor_from_routes(routes)
    sp_lens = 2 * (1 - torch.eye(4, dtype=int))[None]
    nv = count_violations(routes, 4, 2, sp_lens, demand)
    assert nv == 4

    routes = [[0, 1],
              [2, 3],
              [4, 5, 6]]
    routes = get_batch_tensor_from_routes(routes)
    sp_lens = 2 * (1 - torch.eye(7, dtype=int))[None]
    demand = 5 * (1 - torch.eye(7, dtype=int))
    nv = count_violations(routes, 7, 3, sp_lens, demand)
    assert nv == 4 + 12

    routes = [[0, 1],
              [2, 3],
              [4, 5, 6],
              [1 ,3, 5]]
    routes = get_batch_tensor_from_routes(routes)
    nv = count_violations(routes, 7, 4, sp_lens, demand)
    assert nv == 0


def test_cycles():
    routes = [[0, 1, 2, 0, -1]]
    routes = get_batch_tensor_from_routes(routes)
    sp_lens = 2 * (1 - torch.eye(3, dtype=int))[None]
    demand = 5 * (1 - torch.eye(3, dtype=int))
    nv = count_violations(routes, 3, 1, sp_lens, demand)
    assert nv == 1

    sp_lens = 2 * (1 - torch.eye(4, dtype=int))[None]
    demand = 5 * (1 - torch.eye(4, dtype=int))
    nv = count_violations(routes, 4, 1, sp_lens, demand)
    assert nv == 5

    routes = [[0, 1, 2, 3, 0, 3]]
    routes = get_batch_tensor_from_routes(routes)
    nv = count_violations(routes, 4, 1, sp_lens, demand)
    assert nv == 2


def check_oob_route_lens():
    routes = [[0, 1, 2, 0, 3]]
    routes = get_batch_tensor_from_routes(routes)
    demand = 5 * (1 - torch.eye(4, dtype=int))

    nv = count_violations(routes, 4, 1, demand=demand, max_stops=3)
    # one for duplicate, two for oob
    assert nv == 1 + 2

    routes = [[0, 1, 4, 2, 3]]
    routes = get_batch_tensor_from_routes(routes)
    demand = 5 * (1 - torch.eye(5, dtype=int))
    nv = count_violations(routes, 5, 1, demand=demand, max_stops=2)
    # three for oob
    assert nv == 3

    routes = [[0, 1, 2],
              [2, 3, 4, 0, 1, 5, 6],
              [2, 4],
              [1, 5]]
    routes = get_batch_tensor_from_routes(routes)
    demand = 5 * (1 - torch.eye(7, dtype=int))
    nv = count_violations(routes, 7, 4, demand=demand, min_stops=4, 
                          max_stops=5)
    # seven for oob
    assert nv == 1 + 2 + 2 + 2

    routes = [[0, 1, 2],
              [2, 3, 4],
              []]
    routes = get_batch_tensor_from_routes(routes)
    demand = 5 * (1 - torch.eye(5, dtype=int))
    nv = count_violations(routes, 5, 3, demand=demand, min_stops=2, 
                          max_stops=4)
    # two for one route that's too short
    assert nv == 2

def test_n_routes():
    routes = [[0, 1, 2, 3]]
    routes = get_batch_tensor_from_routes(routes)
    sp_lens = 2 * (1 - torch.eye(4, dtype=int))[None]
    demand = 5 * (1 - torch.eye(4, dtype=int))
    nv = count_violations(routes, 4, 2, sp_lens, demand)
    assert nv == 1

    nv = count_violations(routes, 4, 5, sp_lens, demand)
    assert nv == 4

    routes = [[0, 1, 2, 3],
              [3, 2, 1, 0],
              [3, 0, 1]]
    routes = get_batch_tensor_from_routes(routes)
    nv = count_violations(routes, 4, 2, sp_lens, demand)
    assert nv == 1

    routes = [[0, 1, 2, 3],
              [3, 2, 1, 0],
              [3, 0, 1],
              [0, 2]]
    routes = get_batch_tensor_from_routes(routes)
    nv = count_violations(routes, 4, 2, sp_lens, demand)
    assert nv == 2


def test_skipping_stops():
    routes = [[0, 1, 3, 2]]
    routes = get_batch_tensor_from_routes(routes)
    sp_lens = [[0, 2, 3, 4],
               [2, 0, 2, 3],
               [3, 2, 0, 2],
               [4, 3, 2, 0],
               ]
    sp_lens = torch.tensor(sp_lens)[None]
    demand = 5 * (1 - torch.eye(4, dtype=int))
    nv = count_violations(routes, 4, 1, sp_lens, demand)
    assert nv == 1