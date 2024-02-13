# Copyright 2023 Andrew Holliday
# 
# This file is part of the Transit Learning project.
#
# Transit Learning is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.
# 
# Transit Learning is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.
#
# You should have received a copy of the GNU General Public License along with 
# Transit Learning. If not, see <https://www.gnu.org/licenses/>.

import logging as log

import torch
from torch_geometric.loader import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra

from torch_utils import get_batch_tensor_from_routes
from simulation.citygraph_dataset import get_dataset_from_config
from simulation.transit_time_estimator import RouteGenBatchState
import learning.utils as lrnu


def build_route(state, nodes_are_visited, adj_mat, ri, 
                start_at_unvisited=False):
    # sample a random length
    route_len = torch.randint(state.min_route_len.item(), 
                              state.max_route_len.item() + 1, 
                              (1,)).item()
    # sample a random start node
    n_nodes = state.max_n_nodes

    if not nodes_are_visited.any():
        # starting from scratch, so can be any node
        node_start_probs = torch.ones(n_nodes)
    elif start_at_unvisited:
        # must start at an unvisited node
        node_start_probs = (~nodes_are_visited).float()
    else:
        # otherwise, must start at an already-visited node
        node_start_probs = torch.zeros(n_nodes)
        node_start_probs[nodes_are_visited] = 1
    start_node = node_start_probs.multinomial(1).item()

    # grow route randomly until it gets to maximum length
    disconnected = ~nodes_are_visited[start_node]
    route = [start_node]
    just_flipped = False
    while len(route) < route_len:
        # get nodes adjacent to current end
        nodes_are_viable = adj_mat[route[-1]].clone()
        # don't allow nodes that are on this route
        nodes_are_viable[route] = False

        if not nodes_are_viable.any():
            # no way to extend the route from this node!
            if just_flipped:
                # both ends are unviable, so start again from scratch
                just_flipped = False
                route = [start_node]
                log.warning(f'Route {ri} is stuck, starting again')
                continue

            # flip the route and try extending at the other end
            route = list(reversed(route))
            just_flipped = True
            continue
        else:
            # we're going to extend, so reset the flip flag
            just_flipped = False

        if disconnected:
            # try to require that the next node be already covered, to
             # connect this route with the rest of the system
            extra_req = nodes_are_visited
        else:
            # try to require that the next node not already covered, to extend
             # coverage
            extra_req = ~nodes_are_visited

        viable_and_req = nodes_are_viable & extra_req
        if viable_and_req.any():
            # prefer nodes that satisfy the extra requirement
            node_probs = viable_and_req.float()
        else:
            # otherwise, allow any adjacent node
            node_probs = nodes_are_viable.float()

        node = node_probs.multinomial(1).item()
        disconnected &= ~nodes_are_visited[node]
        route.append(node)

    return route
    

def repair(routes, nodes_are_visited, adj_mat, max_route_len):
    dev = adj_mat.device
    shuffled_route_idxs = None
    any_added = None
    n_routes = len(routes)
    while not nodes_are_visited.all():
        if shuffled_route_idxs is None or len(shuffled_route_idxs) == 0:
            # check if things did not improve
            if any_added is not None and not any_added:
                # we're stuck, so return what we've got
                log.warning('Stuck, returning what we have')
                break
        
            # reshuffle the route indices
            shuffled_route_idxs = torch.randperm(n_routes, device=dev)
            # track whether we've added any nodes this time over the routes
            any_added = False
            
        # pop a random route index
        route_idx = shuffled_route_idxs[0]
        shuffled_route_idxs = shuffled_route_idxs[1:]
        route = routes[route_idx]
        if len(route) >= max_route_len:
            # this route is already at maximum length
            continue

        # try to add an unvisited node to the route
        viable_and_unvisited = (~nodes_are_visited) & adj_mat[route[-1]]
        if not viable_and_unvisited.any():
            # reverse the route and try again
            route = list(reversed(route))
            viable_and_unvisited = (~nodes_are_visited) & adj_mat[route[-1]]
        
        if not viable_and_unvisited.any():
            # this route is stuck, so try another one
            continue

        # add a random unvisited node to the route
        node_probs = viable_and_unvisited.float()
        node = node_probs.multinomial(1).item()
        route.append(node)
        routes[route_idx] = route
        nodes_are_visited[node] = True
        any_added = True
    
    return routes

@torch.no_grad()
def build_init_solution(state, *args, **kwargs):
    dev = state.device
    assert state.batch_size == 1
    n_nodes = state.max_n_nodes
    solution = torch.full((state.n_routes_to_plan, n_nodes), -1, 
                          device=dev)    
    # stage 1: generate a set of routes of the right length
    nodes_are_visited = torch.zeros(n_nodes, dtype=torch.bool, device=dev)
    adj_mat = state.street_adj.isfinite()[0].to(dev)
    adj_mat.fill_diagonal_(False)
    routes = []
    for ri in range(state.n_routes_to_plan):
        route = build_route(state, nodes_are_visited, adj_mat, ri)
        
        # keep track of the nodes that have been visited
        nodes_are_visited[torch.tensor(route, device=dev)] = True
        routes.append(route)

    # stage 2: fill in the gaps with the nearest unvisited nodes
    max_route_len = state.max_route_len.item()
    routes = repair(routes, nodes_are_visited, adj_mat, max_route_len)
    
    # while not nodes_are_visited.all():
    #     # drop a random route, plan a new one, and try again
    #     log.warning('Failed to connect all nodes, trying again')
    #     rand_route_idx = torch.randint(len(routes), (1,)).item()
    #     routes.pop(rand_route_idx)

    #     # rebuild the nodes_are_visited mask
    #     nodes_are_visited = torch.zeros(n_nodes, dtype=torch.bool, 
    #                                     device=dev)
    #     for route in routes:
    #         nodes_are_visited[torch.tensor(route, device=dev)] = True

    #     new_route = build_route(state, nodes_are_visited, adj_mat, 
    #                             len(routes), start_at_unvisited=True)
    #     routes.append(new_route)
    #     routes = repair(routes, nodes_are_visited, adj_mat, max_route_len)

    solution = get_batch_tensor_from_routes(routes, device=dev)
    state.add_new_routes(solution)
    return state, None


@hydra.main(version_base=None, config_path="../cfg", config_name="bco_mumford")
def main(cfg: DictConfig):
    global DEVICE
    prefix = 'mumford_init'

    DEVICE, run_name, sum_writer, cost_fn, _ = \
        lrnu.process_standard_experiment_cfg(cfg, prefix, 
                                             weights_required=True)

    # read in the dataset
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    draw = cfg.eval.get('draw', False)
    routes = \
        lrnu.test_method(build_init_solution, test_dl, cfg.eval.n_routes, 
            cfg.eval.min_route_len, cfg.eval.max_route_len, cost_fn, 
            sum_writer=sum_writer, silent=False,
            n_bees=cfg.n_bees, n_iterations=cfg.n_iterations, csv=cfg.eval.csv, 
            device=DEVICE, draw=draw, return_routes=True)[-1]
    if type(routes) is not torch.Tensor:
        routes = get_batch_tensor_from_routes(routes)        
    
    # save the final routes that were produced
    lrnu.dump_routes(run_name, routes.cpu())
    

if __name__ == "__main__":
    main()
