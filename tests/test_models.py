import pytest
import torch
import networkx as nx

from learning.models import update_connections_matrix, \
    check_extensions_add_connections
from torch_utils import floyd_warshall, get_batch_tensor_from_routes


def test_check_extensions_add_connections():
    # check a case were we aren't fully connected and we can add connections
    # no existing connections
    is_linked = torch.eye(3, dtype=bool)[None]
    extensions = torch.full((3, 3, 3), -1, dtype=torch.long)
    extensions[0, 1] = torch.tensor([0, 1, -1])
    extensions[0, 2] = torch.tensor([0, 1, 2])
    extensions[1, 2] = torch.tensor([1, 2, -1])
    extensions[2, 0] = torch.tensor([2, 1, 0])
    extensions[2, 1] = torch.tensor([2, 1, -1])
    extensions[1, 0] = torch.tensor([1, 0, -1])
    extensions = extensions[None]
    add_connections = check_extensions_add_connections(is_linked, extensions)
    assert (add_connections == ~torch.eye(3, dtype=bool)[None]).all()

    # one existing connection
    is_linked[0, 0, 1] = True
    gt_add_connections = ~torch.eye(3, dtype=bool)[None]
    gt_add_connections[0, 0, 1] = False
    add_connections = check_extensions_add_connections(is_linked, extensions)
    assert (add_connections == gt_add_connections).all()

    # two existing connections
    is_linked[0, 2, 1] = True
    gt_add_connections[0, 2, 1] = False
    add_connections = check_extensions_add_connections(is_linked, extensions)
    assert (add_connections == gt_add_connections).all()

    # test batched
    is_linked = is_linked.repeat(5, 1, 1)
    is_linked[1, 0, 1] = False
    is_linked[2, 2, 1] = False
    is_linked[3, 2, 1] = False
    is_linked[3, 0, 1] = False
    is_linked[4, 0, 2] = False
    is_linked[4, 1, 2] = False
    gt_add_connections = gt_add_connections.repeat(5, 1, 1)
    gt_add_connections[1, 0, 1] = True
    gt_add_connections[2, 2, 1] = True
    gt_add_connections[3, 2, 1] = True
    gt_add_connections[3, 0, 1] = True
    gt_add_connections[4, 0, 2] = True
    gt_add_connections[4, 1, 2] = True
    extensions = extensions.repeat(5, 1, 1, 1)
    add_connections = check_extensions_add_connections(is_linked, extensions)
    assert (add_connections == gt_add_connections).all()

    # check case where we're fully connected already
    is_linked = torch.ones((5, 3, 3), dtype=bool)
    no_connection_needed = check_extensions_add_connections(is_linked, 
                                                            extensions)
    assert (no_connection_needed == True).all()


@pytest.mark.parametrize("n_nodes,batch_size", [(10,1), (20,8), (50,16)])
def test_update_connections_matrix(n_nodes, batch_size):
    torch.random.manual_seed(0)
    for _ in range(100):
        batch_unlinked_ems = []
        batch_linked_ems = []
        batch_routes = torch.full((batch_size, n_nodes), -1, dtype=torch.long)
        for bi in range(batch_size):
            graph = nx.erdos_renyi_graph(n_nodes, 0.2)
            edge_matrix = nx.to_numpy_array(graph, nonedge=float('inf'))
            edge_matrix = torch.tensor(edge_matrix)
            edge_matrix.fill_diagonal_(0)
            batch_unlinked_ems.append(edge_matrix.clone())
            # generate a route
            route_len = torch.randint(low=2, high=n_nodes-1, size=(1,))
            route = torch.randperm(n_nodes)[:route_len]
            batch_routes[bi, :route_len] = route
            # add the routes to the edge matrix
            for ii in range(len(route)-1):
                edge_matrix[route[ii], route[ii+1]] = 1
                edge_matrix[route[ii+1], route[ii]] = 1
            batch_linked_ems.append(edge_matrix)

        # concatenate the edge matrices and routes
        batch_unlinked_ems = torch.stack(batch_unlinked_ems)
        # run floyd-warshall on the edge matrices
        _, dists = floyd_warshall(batch_unlinked_ems)
        is_linked = dists.isfinite()
        # run update_connections_matrix() on output from floyd-warshall
        updated_is_linked = update_connections_matrix(is_linked, batch_routes, 
                                                      True)
        batch_linked_ems = torch.stack(batch_linked_ems)
        # run floyd-warshall on output from update_connections_matrix()
        _, updated_dists = floyd_warshall(batch_linked_ems)
        gt_updated_is_linked = updated_dists.isfinite()
        # this variable is useful to inspect if the below assertion fails
        wheres = torch.where(updated_is_linked != gt_updated_is_linked)
        assert (updated_is_linked == gt_updated_is_linked).all()


