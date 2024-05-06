import logging as log

import torch
import networkx as nx

from torch_utils import get_batch_tensor_from_routes, load_routes_tensor, \
    aggr_edges_over_sequences, reconstruct_all_paths
from simulation.transit_time_estimator import RouteGenBatchState


def init_from_cfg(state, init_cfg):
    """Initializes the network according to the configuration.
    
    Args:
        state: The initial state of the system, not including the routes.
        init_cfg: The configuration for the initialization method.
    
    Returns:
        The initial network.
    """
    if init_cfg is None:
        # return torch.zeros((0, 0, 0), device=state.device, dtype=torch.long)
        return None
    elif init_cfg.method == 'load':
        return load_routes_tensor(init_cfg.path, state.device)
    elif init_cfg.method == 'john':
        alpha = init_cfg.get('alpha', state.alpha)
        return john_init(state, alpha, init_cfg.prioritize_direct_connections)
    elif init_cfg.method == 'nikolic':
        return nikolic_init(state)
    else:
        raise ValueError(f'Unknown initialization method: {init_cfg.method}')
    # TODO bring in hyperheuristic initialization scheme


def john_init(batch_state: RouteGenBatchState, alpha: float, 
              prioritize_direct_connections: bool = True):
    """Constructs a network based on the algorithm of John et al. (2014).
    
    Args:
        state: A representation of the city and the problem constraints.
        alpha: Controls trade-off between transit time and demand. alpha=1
            means only demand matters, alpha=0 means only transit time matters.
    """
    assert batch_state.symmetric_routes, 'John et al. (2014) requires symmetric routes'

    # compute weights for all of the edges
    norm_times = batch_state.drive_times / batch_state.drive_times.max()
    norm_demand = batch_state.demand / batch_state.demand.max()
    beta = 1 - alpha
    all_edge_costs = beta * norm_times + alpha * (1 - norm_demand)
    have_edges = batch_state.street_adj.isfinite() & \
        (batch_state.street_adj > 0)
    all_edge_costs[~have_edges] = float('inf')

    if batch_state.batch_size > 1:
        states = batch_state.unbatch()
    else:
        states = [batch_state]

    # build a spanning sub-graph of routes
    networks = []
    for edge_costs, state in zip(all_edge_costs, states):
        network = []
        has_edges = state.street_adj.isfinite() & (state.street_adj > 0)
        has_edges = has_edges.squeeze(0)
        n_nodes = state.nodes_per_scenario[0].item()
        are_visited = torch.zeros(n_nodes, dtype=torch.bool, 
                                  device=state.device)
        are_directly_connected = torch.eye(n_nodes, dtype=torch.bool, 
                                           device=state.device)

        # While all nodes are not yet visited:
        while not are_visited.all():
            # select a "seed pair" of vertices to be the start of the route
            if len(network) == 0:
                # first iteration, we can choose any edge
                scores = edge_costs
            else:
                # choose an edge between a covered and an uncovered node
                connects_new = are_visited[:, None] & ~are_visited
                # add the max edge cost to pairs that don't connect a new node,
                 # so they can't get chosen by the argmin later
                scores = edge_costs.clone()
                scores[~connects_new] = edge_costs.max() + 1
                # scores = edge_costs + ~connects_new * (edge_costs.max() + 1)

            flat_seed_pair = torch.argmin(scores).item()
            seed_pair = (flat_seed_pair // n_nodes,
                         flat_seed_pair % n_nodes)
            route = list(seed_pair)
            are_on_route = torch.zeros(n_nodes, dtype=torch.bool,
                                       device=state.device)
            are_on_route[[seed_pair]] = True
            are_visited[[seed_pair]] = True

            # expand the seed pair to form a route by adding adjacent vertices
            while len(route) < state.max_route_len:
                # find valid extension nodes
                are_valid_start_exts = has_edges[route[0]] & ~are_on_route
                are_valid_end_exts = has_edges[route[-1]] & ~are_on_route
                
                # check if any of them are not yet visited at all
                are_valid_exts = are_valid_start_exts | are_valid_end_exts
                if not are_valid_exts.any():
                    # no valid extensions, so we're done
                    break

                if (are_valid_exts & ~are_visited).any():
                    # we can visit an unvisited node, so we must do so
                    are_valid_start_exts &= ~are_visited
                    are_valid_end_exts &= ~are_visited
                
                # choose the next node to add by least edge cost
                valid_start_exts = torch.nonzero(are_valid_start_exts).squeeze(-1)
                valid_end_exts = torch.nonzero(are_valid_end_exts).squeeze(-1)
                start_edge_costs = edge_costs[route[0], valid_start_exts]
                end_edge_costs = edge_costs[route[-1], valid_end_exts]
                ext_edge_costs = torch.cat((start_edge_costs, end_edge_costs))
                min_edge_cost = ext_edge_costs.min()
                options = ext_edge_costs == min_edge_cost
                # break ties randomly
                chosen_ext_loc = options.float().multinomial(1).item()
                if chosen_ext_loc < len(start_edge_costs):
                    chosen_node = valid_start_exts[chosen_ext_loc].item()
                    route.insert(0, chosen_node)
                else:
                    end_loc = chosen_ext_loc - len(start_edge_costs)
                    chosen_node = valid_end_exts[end_loc].item()
                    route.append(chosen_node)
                
                # mark the node as on the route and visited
                are_on_route[chosen_node] = True
                are_visited[chosen_node] = True

            # route is finished
            if len(route) >= state.min_route_len:
                # route is long enough to add
                route = torch.tensor(route)
                # mark the route's nodes as directly connected
                are_directly_connected[route, route] = True
                network.append(route)

            if len(network) == state.n_routes_to_plan:
                # we have enough routes, so we're done
                break

        if not are_visited.all() and len(network) == state.n_routes_to_plan:
            raise RuntimeError('Failed to span the graph')

        if len(network) < state.n_routes_to_plan:
            # Use approach of Shih and Mahmassani to add more routes
            # sort node pairs by descending demand
            if prioritize_direct_connections:
                # only consider node pairs that are not directly connected
                nodepairs = torch.nonzero(~are_directly_connected)
                pair_demands = state.demand[0, ~are_directly_connected]
            else:
                pair_demands = state.demand[0]
                nodepairs = torch.nonzero(torch.ones_like(pair_demands))

            sorted_indices = pair_demands.flatten().argsort(descending=True)
            sorted_pairs = nodepairs[sorted_indices]

            # build a networkx graph
            adj_mat = state.street_adj[0].cpu()
            adj_mat[adj_mat.isinf()] = 0
            graph = nx.from_numpy_matrix(adj_mat.numpy())

            # add the existing routes to the graph to get the transit times
            state.add_new_routes([network])

            # For each pair in order, until $|\mathcal{R}| = S$:
            for (src, dst) in sorted_pairs:
                # Use Yen's k-shortest path algorithm (see ref. 22) with k=10 
                 # to see if a valid route between the nodes exists that 
                 # doesn't violate any constraints
                paths = yen_k_shortest_paths(graph, src.item(), dst.item(), 10)
                for path in paths:
                    # check if the path is valid
                    if state.min_route_len <= len(path) <= state.max_route_len \
                        and path not in network:
                        # it is valid and shorter than the shortest 
                         # transit path between the nodes, add it to 
                         # |\mathcal{R}|
                        path_len = nx.path_weight(graph, path, weight='weight')
                        if path_len < state.transit_times[0, src, dst]:
                            # add the path and move on to the next node pair
                            path = torch.tensor(path)
                            network.append(path)
                            state.add_new_routes([[path]])
                            break                

                if len(network) == state.n_routes_to_plan:
                    # we have enough routes, so we're done
                    break

        if len(network) < state.n_routes_to_plan:
            raise RuntimeError('Failed to find enough routes')

        networks.append(network)

    # make sure changes to state don't affect the original state
    batch_state.clear_routes()
    networks = get_batch_tensor_from_routes(networks, state.device)

    return networks


def yen_k_shortest_paths(graph: nx.Graph, src_index: int, dst_index: int, 
                         kk: int):
    """Yen's k-shortest paths algorithm for a graph.
    
    Args:
        graph: The graph to find paths in.
        src_index: The index of the source node.
        dst_index: The index of the destination node.
    
    Returns:
        A list of the k shortest paths from the source to the destination.
    """
    shortest = nx.dijkstra_path(graph, src_index, dst_index)
    k_shortest = [shortest]
    prev_path = shortest

    potential_paths = []
    ptnl_path_costs = []
    for _ in range(kk - 1):
        prev_path = k_shortest[-1]
        for jj in range(len(prev_path) - 1):
            # spur node is the j-th node in the previous path
            spur_node = prev_path[jj]
            root_path = prev_path[:jj + 1]

            graph_copy = graph.copy()
            for path in k_shortest:
                if root_path == path[:jj + 1]:
                    # remove the links that are part of the previous shortest 
                     # paths which share the same root path, so they won't
                     # be taken again
                    try:
                        graph_copy.remove_edge(path[jj], path[jj + 1])
                    except nx.NetworkXError:
                        # the edge has already been removed. that's fine.
                        pass
            
            # remove nodes before the spur node so the next shortest path won't
             # go along them
            for node in root_path:
                if node != spur_node:
                    graph_copy.remove_node(node)
            
            # calculate the spur path from the spur node to the destination
            try:
                spur_path = nx.dijkstra_path(graph_copy, spur_node, dst_index)
            except nx.NetworkXNoPath:
                continue
            
            # the root path followed by the spur path is another potential path
            total_path = root_path[:-1] + spur_path
            if total_path not in potential_paths:
                potential_paths.append(total_path)
                ptnl_path_costs.append(nx.path_weight(graph, total_path, 
                                                      weight='weight'))

        if not potential_paths:
            # there are no remaining spur paths to try
            break

        # choose the lowest cost path
        min_cost = min(ptnl_path_costs)
        min_cost_index = ptnl_path_costs.index(min_cost)
        path = potential_paths.pop(min_cost_index)
        ptnl_path_costs.pop(min_cost_index)
        k_shortest.append(path)

    return k_shortest


def nikolic_init(state: RouteGenBatchState):
    """Constructs a network based on the algorithm of Nikolic and Teodorovic 
        (2013).
    """
    shortest_paths, _ = reconstruct_all_paths(state.nexts)

    dev = shortest_paths.device
    batch_size = shortest_paths.shape[0]
    batch_idxs = torch.arange(batch_size, device=dev)
    max_n_nodes = state.max_n_nodes
    dm_uncovered = state.demand.clone()
    # add dummy column and row
    dm_uncovered = torch.nn.functional.pad(dm_uncovered, (0, 1, 0, 1))
    n_routes = state.n_routes_to_plan
    best_networks = torch.full((batch_size, n_routes, max_n_nodes), -1, 
                                device=dev)
    log.info('computing initial network')
    # stop-to-itself routes are all invalid
    terms_are_invalid = torch.eye(max_n_nodes + 1, device=dev, dtype=bool)
    # routes to and from the dummy stop are invalid
    terms_are_invalid[max_n_nodes, :] = True
    terms_are_invalid[:, max_n_nodes] = True
    terms_are_invalid = terms_are_invalid[None].repeat(batch_size, 1, 1)
    for ri in range(n_routes):
        # compute directly-satisfied-demand matrix
        direct_sat_dmd = get_direct_sat_dmd(dm_uncovered, shortest_paths,
                                            state.symmetric_routes)
        # set invalid term pairs to -1 so they won't be selected even if there
         # is no uncovered demand.
        direct_sat_dmd[terms_are_invalid] = -1

        # choose maximum DS route and add it to the initial network
        flat_dsd = direct_sat_dmd.flatten(1, 2)
        _, best_flat_idxs = flat_dsd.max(dim=1)
        # add 1 to account for the dummy column and row
        best_i = torch.div(best_flat_idxs, (max_n_nodes + 1), 
                           rounding_mode='floor')
        best_j = best_flat_idxs % (max_n_nodes + 1)
        # batch_size x route_len
        routes = shortest_paths[batch_idxs, best_i, best_j]
        best_networks[:, ri, :routes.shape[-1]] = routes

        # mark new routes as in use
        terms_are_invalid[batch_idxs, best_i, best_j] = True
        if state.symmetric_routes:
            terms_are_invalid[batch_idxs, best_j, best_i] = True

        # remove newly-covered demand from uncovered demand matrix
        for ii in range(routes.shape[-1] - 1):
            cur_stops = routes[:, ii][:, None]
            later_stops = routes[:, ii+1:]
            dm_uncovered[batch_idxs[:, None], cur_stops, later_stops] = 0
            if state.symmetric_routes:
                maxidx = routes.shape[-1] - 1
                cur_stops = routes[:, maxidx - ii]
                later_stops = routes[:, maxidx - ii - 1]
                dm_uncovered[batch_idxs[:, None], later_stops, cur_stops] = 0

    return best_networks


def get_direct_sat_dmd(stop_dmd, shortest_paths, symmetric):
    direct_sat_dmd = torch.zeros_like(stop_dmd)
    summed_demands = aggr_edges_over_sequences(shortest_paths, 
                                               stop_dmd[..., None])
    if symmetric:
        summed_demands += \
            aggr_edges_over_sequences(shortest_paths.transpose(1,2),
                                      stop_dmd.transpose(1,2)[..., None])
    direct_sat_dmd[:, :-1, :-1] = summed_demands.squeeze(-1)

    # assumes there's a dummy column and row
    direct_sat_dmd[:, -1] = 0
    direct_sat_dmd[:, :, -1] = 0
    return direct_sat_dmd
