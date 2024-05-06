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
import copy

import torch
from torch_geometric.data import Batch, HeteroData
from dataclasses import dataclass
from typing import Optional

from simulation.citygraph_dataset import STOP_KEY
from torch_utils import floyd_warshall, reconstruct_all_paths, \
    get_batch_tensor_from_routes, get_route_edge_matrix, get_route_leg_times, \
    aggregate_edge_features


MEAN_STOP_TIME_S = 60
AVG_TRANSFER_WAIT_TIME_S = 300
UNSAT_PENALTY_EXTRA_S = 3000


def enforce_correct_batch(matrix, batch_size):
    if matrix.ndim == 2:
        matrix = matrix[None]
    if matrix.shape[0] > 1:
        assert batch_size == matrix.shape[0]
    elif batch_size > 1:
        shape = (batch_size,) + (-1,) * (matrix.ndim - 1)
        matrix = matrix.expand(*shape)
    return matrix


class ExtraStateData(HeteroData):
    """A class for holding data, some of it computed, that is specific to one
    scenario in a state."""
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key in ['base_valid_terms_mat', 
                   'valid_terms_mat',
                   'mean_stop_time',
                   'fixed_routes_file',
                   'transfer_time_s',
                   'total_route_time', 
                   'n_routes_to_plan', 
                   'min_route_len',
                   'max_route_len', 
                   'n_nodes_in_scenario',
                   'directly_connected', 
                   'route_mat',
                   'route_nexts',
                   'transit_times',
                   'has_path', 
                   'n_transfers',
                   'norm_node_features',
                   'fixed_routes']:
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class RouteGenBatchState:
    def __init__(self, graph_data, cost_obj, n_routes_to_plan, min_route_len=2,
                 max_route_len=None, valid_terms_mat=None, cost_weights=None,
                 fixed_routes=None):
        # do initialization needed to make properties work
        if not isinstance(graph_data, Batch):
            if not isinstance(graph_data, list):
                graph_data = [graph_data]
            graph_data = Batch.from_data_list(graph_data)

        # set members that live directly on this object
        self.graph_data = graph_data
        # right now this must have the same value for all scenarios
        self.n_routes_to_plan = n_routes_to_plan
        self.symmetric_routes = cost_obj.symmetric_routes
        self.is_demand_float = (self.graph_data.demand > 0).to(torch.float32)

        # the object isn't ready to give this property yet, so find it here
        dev = graph_data[STOP_KEY].x.device
        max_n_nodes = max([dd.num_nodes for dd in graph_data.to_data_list()])
        if valid_terms_mat is None:
            # all terminal pairs (i,j) are valid except if i = j
            valid_terms_mat = ~torch.eye(max_n_nodes, device=dev, dtype=bool)
            valid_terms_mat = valid_terms_mat.repeat(self.batch_size, 1, 1)

        if cost_weights is None:
            # get the cost weights
            cost_weights = cost_obj.get_weights(device=dev)
        for key, val in cost_weights.items():
            # expand the cost weights to match the batch
            if type(val) is not torch.Tensor:
                val = torch.tensor([val], device=dev)
            if val.numel() == 1:
                val = val.expand(graph_data.num_graphs)
            cost_weights[key] = val

        extra_datas = []
        for ii, dd in enumerate(graph_data.to_data_list()):
            extra_data = ExtraStateData()
            extra_data.route_mat = \
                torch.full((1, max_n_nodes, max_n_nodes), float('inf'), 
                           device=self.device)
            extra_data.transit_times = extra_data.route_mat.clone()
            dircon = torch.eye(max_n_nodes, device=self.device, dtype=bool)

            extra_data.directly_connected = dircon[None]
            extra_data.has_path = extra_data.directly_connected.clone()
            extra_data.base_valid_terms_mat = valid_terms_mat[ii]
            extra_data.valid_terms_mat = valid_terms_mat[ii]
            extra_data.mean_stop_time = \
                torch.tensor(cost_obj.mean_stop_time_s, device=dev)
            extra_data.transfer_time_s = \
                torch.tensor(cost_obj.avg_transfer_wait_time_s, device=dev)
            extra_data.total_route_time = torch.zeros((), device=dev)
            # make this a tensor so it's stackable
            extra_data.n_nodes_in_scenario = torch.tensor(dd.num_nodes,
                                                          device=dev)
            if isinstance(min_route_len, torch.Tensor):
                if min_route_len.numel() == 1:
                    extra_data.min_route_len = min_route_len
                else:
                    extra_data.min_route_len = min_route_len[ii]
            else:
                extra_data.min_route_len = torch.tensor(min_route_len,
                                                        device=dev)
            extra_data.min_route_len.squeeze_()

            if max_route_len is None:
                max_route_len = dd.num_nodes
            if isinstance(max_route_len, torch.Tensor):
                if max_route_len.numel() == 1:
                    extra_data.max_route_len = max_route_len
                else:
                    extra_data.max_route_len = max_route_len[ii]
            else:
                extra_data.max_route_len = torch.tensor(max_route_len,
                                                        device=dev)
            extra_data.max_route_len.squeeze_()

            extra_data.cost_weights = {}
            for key, val in cost_weights.items():
                # expand the cost weights to match the batch
                extra_data.cost_weights[key] = val[ii]

            extra_datas.append(extra_data)
            if fixed_routes is not None:
                # fixed_routes must be the same for all instances in the batch
                assert fixed_routes.shape[0] == 1
                extra_data.fixed_routes = fixed_routes[0]
            else:
                extra_data.fixed_routes = torch.zeros(0)

        self.extra_data = Batch.from_data_list(extra_datas)

        # this initializes route data
        self.clear_routes()

    def replace_routes(self, batch_new_routes, 
                       only_routes_with_demand_are_valid=False, 
                       invalid_directly_connected=False):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
        self.add_new_routes(batch_new_routes, 
                            only_routes_with_demand_are_valid,
                            invalid_directly_connected)

    def clear_routes(self):
        self._clear_routes_helper()
        if self.extra_data.fixed_routes.numel() > 0:
            self._add_routes_to_tensors(self.extra_data.fixed_routes)
        self._update_route_data()

    def _clear_routes_helper(self):
        directly_connected = torch.eye(self.max_n_nodes, device=self.device, 
                                       dtype=bool)
        self.extra_data.directly_connected = \
            directly_connected.repeat(self.batch_size, 1, 1)
        self.routes = [[] for _ in range(self.batch_size)]        
        self.extra_data.route_mat = \
            torch.full((self.batch_size, self.max_n_nodes, self.max_n_nodes), 
                        float('inf'), device=self.device)
        self.extra_data.valid_terms_mat = \
            self.extra_data.base_valid_terms_mat.clone()
        self.extra_data.total_route_time[...] = 0

    def add_new_routes(self, batch_new_routes,
                       only_routes_with_demand_are_valid=False, 
                       invalid_directly_connected=False):
        self._add_routes_to_tensors(batch_new_routes, 
                                   only_routes_with_demand_are_valid, 
                                   invalid_directly_connected)
        self._add_routes_to_list(batch_new_routes)

    def _add_routes_to_tensors(self, batch_new_routes,
                               only_routes_with_demand_are_valid=False, 
                               invalid_directly_connected=False):
        """Takes a tensor of new routes. The first dimension is the batch"""
        # incorporate new routes into the route graphs
        if type(batch_new_routes) is list:
            batch_new_routes = get_batch_tensor_from_routes(batch_new_routes,
                                                            self.device)
        # add new routes to the route matrix.
        new_route_mat = get_route_edge_matrix(
            batch_new_routes, self.drive_times, 
            self.mean_stop_time, self.symmetric_routes)
        self.extra_data.route_mat = \
            torch.minimum(self.route_mat, new_route_mat)
        route_lengths = (batch_new_routes > -1).sum(dim=-1)

        batch_idxs, route_froms, route_tos = \
            torch.where(self.route_mat < float('inf'))
        _, counts = torch.unique(batch_idxs, return_counts=True)
        counts = counts.tolist()
        edge_idxs = torch.stack((route_froms, route_tos))
        edge_times = self.route_mat[batch_idxs, route_froms, route_tos]

        for bi in range(self.batch_size):
            count = counts[0]
            counts = counts[1:]
            route_idxs = edge_idxs[:, :count]
            edge_idxs = edge_idxs[:, count:]
            # route_data.edge_index = route_idxs
            # route_data.edge_attr = edge_times[:count][:, None]
            edge_times = edge_times[count:]
            self.directly_connected[bi, route_idxs[0], route_idxs[1]] = True

        # allow connection to any node 'upstream' of a demand dest, or
         # 'downstream' of a demand src.
        if only_routes_with_demand_are_valid:
            connected_T = self.nodes_are_connected(2).transpose(1, 2)
            connected_T = connected_T.to(torch.float32)
            valid_upstream = self.is_demand_float.bmm(connected_T)
            self.extra_data.valid_terms_mat[valid_upstream.to(bool)] = True
            valid_downstream = connected_T.bmm(self.is_demand_float)
            self.extra_data.valid_terms_mat[valid_downstream.to(bool)] = True

        if invalid_directly_connected:
            self.extra_data.valid_terms_mat[self.directly_connected] = False

        leg_times = get_route_leg_times(batch_new_routes, 
                                        self.graph_data.drive_times,
                                        self.mean_stop_time)
        total_new_time = leg_times.sum(dim=(1,2))

        if self.symmetric_routes:
            transpose_dtm = self.graph_data.drive_times.transpose(1, 2)
            return_leg_times = get_route_leg_times(batch_new_routes, 
                                                   transpose_dtm,
                                                   self.mean_stop_time)
            total_new_time += return_leg_times.sum(dim=(1,2))
            self.extra_data.valid_terms_mat = self.valid_terms_mat & \
                self.valid_terms_mat.transpose(1, 2)
        
        self.extra_data.total_route_time += total_new_time
        self._update_route_data()

    def _add_routes_to_list(self, batch_routes):
        for bi in range(self.batch_size):
            for route in batch_routes[bi]:
                if type(route) is list:
                    route = torch.tensor(route, device=self.device)
                length = (route > -1).sum()
                if length <= 1:
                    # this is an invalid route
                    log.warn('invalid route!')
                    continue
                self.routes[bi].append(route[:length])

    def _update_route_data(self):
        # do things that have to be done whether we added or removed routes
        nexts, transit_times = floyd_warshall(self.route_mat)
        self.extra_data.route_nexts = nexts
        self.extra_data.transit_times = transit_times
        self.extra_data.has_path = transit_times < float('inf')
        _, path_lens = reconstruct_all_paths(nexts)
        # number of transfers is number of nodes except start and end
        n_transfers = (path_lens - 2).clamp(min=0)
        # set number of transfers where there is no path to 0
        n_transfers[~self.has_path] = 0
        transfer_penalties = n_transfers * self.transfer_time_s[:, None, None]
        self.extra_data.transit_times += transfer_penalties
        self.extra_data.n_transfers = n_transfers

    def set_normalized_features(self, norm_stop_features):
        self.extra_data.norm_node_features = norm_stop_features

    def unbatch(self):
        """return a list of states, one for each scenario in the batch."""
        if self.batch_size == 1:
            return [self]
        
        states = []
        graph_datas = self.graph_data.to_data_list()
        extra_datas = self.extra_data.to_data_list()
        for gd, ed in zip(graph_datas, extra_datas):
            # start with a shallow copy to keep the parts that aren't specific
             # to any batch element
            substate = copy.copy(self)
            substate.graph_data = Batch.from_data_list([gd])
            substate.extra_data = Batch.from_data_list([ed])
            states.append(substate)

        return states
    
    def clone(self):
        """return a deep copy of this state."""
        return copy.deepcopy(self)
    
    @staticmethod
    def batch_from_list(state_list):
        """return a batch state from a list of states."""
        if len(state_list) == 1:
            return state_list[0]
        
        graph_datas = sum([ss.graph_data.to_data_list() for ss in state_list],
                          [])
        extra_datas = sum([ss.extra_data.to_data_list() for ss in state_list], 
                          [])
        batch_graph_data = Batch.from_data_list(graph_datas)
        batch_extra_data = Batch.from_data_list(extra_datas)
        batch_state = copy.copy(state_list[0])
        batch_state.graph_data = batch_graph_data
        batch_state.extra_data = batch_extra_data
        batch_state.routes = [[] for _ in state_list]
        return batch_state
    
    def get_global_state_features(self):
        cost_weights = self.cost_weights_tensor
        avg_route_time = self.total_route_time / self.n_routes_to_plan

        so_far = self.n_routes_so_far
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        n_routes_log_feats = (both + 1).log()
        # use fractions so it's independent of n_routes_to_plan
        n_routes_frac_feats = both / (so_far + left)[:, None]

        n_disconnected_demand_edges = self.get_n_disconnected_demand_edges()
        # as with n_routes feats, use both log and fractional
        log_uncovered = (n_disconnected_demand_edges + 1).log()
        frac_uncovered = n_disconnected_demand_edges / self.n_demand_edges
        uncovered_feats = torch.stack((log_uncovered, 
                                       frac_uncovered
                                     ), dim=-1)
        # curr_route_n_stops = self.current_route_n_stops[:, None]

        served_demand = (self.has_path * self.demand).sum(dim=(1, 2))
        # total_demand = self.demand.sum(dim=(1, 2))
        # unserved_demand = total_demand - served_demand
        # log_unserved = (unserved_demand + 1).log()
        tt = self.transit_times.clone()
        tt[~self.has_path] = 0
        total_demand_time = (self.demand * tt).sum(dim=(1,2))
        mean_demand_time = total_demand_time / (served_demand + 1e-6)
        denom = self.drive_times.mean((1, 2))
        mean_demand_time_norm = mean_demand_time / denom

        global_features = torch.cat((
            cost_weights, avg_route_time[:, None], n_routes_log_feats, 
            n_routes_frac_feats, 
            uncovered_feats, # curr_route_n_stops,
            mean_demand_time_norm[:, None], # log_unserved[:, None]
        ), dim=-1)
        return global_features

    def get_n_disconnected_demand_edges(self):
        # count the number of demand edges that are disconnected
        nopath = ~self.has_path
        needed_path_missing = nopath & (self.demand > 0)
        n_disconnected_demand_edges = needed_path_missing.sum(dim=(1, 2))
        if self.symmetric_routes:
            # connecting one of the two connects both, so count each 2 as 
             # just 1.
            n_disconnected_demand_edges = n_disconnected_demand_edges / 2
        return n_disconnected_demand_edges

    def get_shortest_path_sequences(self):
        if self.extra_data.shortest_path_sequences.numel() == 0:
            path_seqs, _ = reconstruct_all_paths(self.nexts)
            self.extra_data.shortest_path_sequences = path_seqs
        else:
            path_seqs = self.extra_data.shortest_path_sequences
        return path_seqs

    def to_device(self, device):
        dev_state = self.clone()
        dev_state.graph_data = dev_state.graph_data.to(device)
        dev_state.extra_data = dev_state.extra_data.to(device)
        for ii in range(self.batch_size):
            for jj, route in dev_state.routes[ii]:
                dev_state.routes[ii][jj] = route.to(device)
         
        dev_state.is_demand_float = dev_state.is_demand_float.to(device)
        
        return dev_state
        
    @property
    def n_demand_edges(self):
        n_demand_edges = (self.demand > 0).sum(dim=(1, 2))
        if self.symmetric_routes:
            n_demand_edges = (n_demand_edges / 2).ceil()
        return n_demand_edges

    @property
    def cost_weights_tensor(self):
        cost_weights_list = []
        for key in sorted(self.cost_weights.keys()):
            if type(self.cost_weights[key]) is torch.Tensor:
                cw = self.cost_weights[key].to(self.device)
                if cw.ndim == 0:
                    cw = cw[None]
            else:
                cw = torch.tensor(self.cost_weights[key], 
                                  device=self.device)[None]

            cost_weights_list.append(cw)
        cost_weights = torch.stack(cost_weights_list, dim=1)
        if cost_weights.shape[0] == 1:
            cost_weights = cost_weights.expand(self.batch_size, -1)
        if cost_weights.shape[0] > self.batch_size:
            cost_weights = cost_weights[:self.batch_size]
        return cost_weights
    
    @property
    def node_covered_mask(self):
        have_out_paths = self.directly_connected.any(dim=1)
        if self.symmetric_routes:
            are_covered = have_out_paths
        else:
            are_covered = have_out_paths & self.directly_connected.any(dim=2)
        return are_covered

    # don't expose the extra data directly, just provide this interface.
    @property
    def norm_node_features(self):
        if hasattr(self.extra_data, 'norm_node_features'):
            return self.extra_data.norm_node_features
        else:
            return None
        
    @property
    def norm_cost_weights(self):
        if hasattr(self.extra_data, 'norm_cost_weights'):
            return self.extra_data.norm_cost_weights
        else:
            return None
    
    @property
    def valid_terms_mat(self):
        return self.extra_data.valid_terms_mat
    
    @property
    def mean_stop_time(self):
        return self.extra_data.mean_stop_time
    
    @property
    def transfer_time_s(self):
        return self.extra_data.transfer_time_s

    @property
    def total_route_time(self):
        return self.extra_data.total_route_time

    @property
    def min_route_len(self):
        return self.extra_data.min_route_len
    
    @property
    def max_route_len(self):
        return self.extra_data.max_route_len
    
    @property
    def nodes_per_scenario(self):
        return self.extra_data.n_nodes_in_scenario
    
    @property
    def alpha(self):
        return self.extra_data.cost_weights['demand_time_weight']

    @property
    def cost_weights(self):
        return self.extra_data.cost_weights
    
    @property
    def directly_connected(self):
        return self.extra_data.directly_connected

    @property
    def nexts(self):
        return self.graph_data.nexts

    @property
    def route_mat(self):
        return self.extra_data.route_mat
    
    @property
    def route_nexts(self):
        return self.extra_data.route_nexts
    
    @property
    def transit_times(self):
        return self.extra_data.transit_times
    
    @property
    def has_path(self):
        return self.extra_data.has_path
    
    @property
    def n_transfers(self):
        return self.extra_data.n_transfers

    @property
    def street_adj(self):
        return self.graph_data.street_adj

    @property
    def demand(self):
        return self.graph_data.demand

    @property
    def drive_times(self):
        return self.graph_data.drive_times

    @property
    def n_routes_so_far(self):
        nrsf = len(self.routes[0])
        return torch.full((self.batch_size,), nrsf, dtype=torch.float32,
                          device=self.device)
    
    @property
    def batch_size(self):
        return self.graph_data.num_graphs

    @property
    def n_routes_left_to_plan(self):
        return self.n_routes_to_plan - self.n_routes_so_far
    
    def get_n_routes_features(self):
        so_far = self.n_routes_so_far
        left = self.n_routes_left_to_plan
        both = torch.stack((so_far, left), dim=-1)
        return (both + 1).log()

    def nodes_are_connected(self, n_transfers=2):
        dircon_float = self.directly_connected.to(torch.float32)
        connected = dircon_float
        for _ in range(n_transfers):
            # connected by 2 or fewer transfers
            connected = connected.bmm(dircon_float)
        return connected.bool()

    @property
    def device(self):
        return self.graph_data[STOP_KEY].x.device
    
    @property
    def max_n_nodes(self):
        return max(self.nodes_per_scenario)
    
    @property
    def route_n_stops(self):
        route_n_stops = [[len(rr) for rr in br] for br in self.routes]
        return torch.tensor(route_n_stops, device=self.device)


@dataclass
class CostHelperOutput:
    total_demand_time: torch.Tensor
    total_route_time: torch.Tensor
    trips_at_transfers: torch.Tensor
    total_demand: torch.Tensor
    unserved_demand: torch.Tensor
    total_transfers: torch.Tensor
    trip_times: torch.Tensor
    n_disconnected_demand_edges: torch.Tensor
    n_stops_oob: torch.Tensor
    batch_routes: torch.Tensor
    per_route_riders: Optional[torch.Tensor] = None
    cost: Optional[torch.Tensor] = None

    @property
    def mean_demand_time(self):
        served_demand = self.total_demand - self.unserved_demand
        # avoid division by 0
        return self.total_demand_time / (served_demand + 1e-6)

    def get_metrics(self):
        """return a dictionary with the metrics we usually report."""
        frac_tat = self.trips_at_transfers / self.total_demand[:, None]
        percent_tat = frac_tat * 100
        metrics = {
            'cost': self.cost,
            'ATT': self.mean_demand_time / 60,
            'RTT': self.total_route_time / 60,
            '$d_0$': percent_tat[:, 0],
            '$d_1$': percent_tat[:, 1],
            '$d_2$': percent_tat[:, 2],
            '$d_{un}$': percent_tat[:, 3],
            '# disconnected node pairs': 
                self.n_disconnected_demand_edges.float(),
            '# stops out of bounds': self.n_stops_oob.float(),
        }
        return metrics

    def get_metrics_tensor(self):
        """return a tensor with the metrics we usually report."""
        metrics = self.get_metrics()
        metrics = torch.stack([metrics[k] for k in metrics], dim=-1)
        return metrics
    

class CostModule(torch.nn.Module):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False):
        super().__init__()
        self.mean_stop_time_s = mean_stop_time_s
        self.avg_transfer_wait_time_s = avg_transfer_wait_time_s
        self.symmetric_routes = symmetric_routes
        self.low_memory_mode = low_memory_mode

    def get_metric_names(self):
        dummy_obj = CostHelperOutput(
            torch.zeros(1), torch.zeros(1), torch.zeros(1, 4), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1), torch.zeros(1),
            torch.zeros(1), torch.zeros(1), torch.zeros(1))
        return dummy_obj.get_metrics().keys()

    def _cost_helper(self, state, return_per_route_riders=False):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        log.debug("formatting tensors")
        drive_times_matrix = state.drive_times
        demand_matrix = state.demand
        dev = drive_times_matrix.device

        log.debug("assembling route edge matrices")
        # assemble route graph
        batch_routes = get_batch_tensor_from_routes(state.routes, dev)
        route_lens = (batch_routes > -1).sum(-1)

        log.debug("summing cost components")

        nopath = ~state.has_path
        needed_path_missing = nopath & (demand_matrix > 0)
        n_disconnected_demand_edges = needed_path_missing.sum(dim=(1, 2))

        zero = torch.zeros_like(route_lens)
        route_len_delta = (self.min_route_len - route_lens).maximum(zero)
        # don't penalize placeholer "dummy" routes in the tensor
        route_len_delta[route_lens == 0] = 0
        if self.max_route_len is not None:
            route_len_over = (route_lens - self.max_route_len).maximum(zero)
            route_len_delta = route_len_delta + route_len_over
        n_stops_oob = route_len_delta.sum(-1)

        # calculate the amount of demand at each number of transfers
        trips_at_transfers = torch.zeros(state.batch_size, 4, device=dev)
        # trips with no path get '3' transfers so they'll be included in d_un,
         # not d_0
        n_transfers = state.n_transfers.clone()
        n_transfers[nopath] = 3
        for ii in range(3):
            d_i = (demand_matrix * (n_transfers == ii)).sum(dim=(1, 2))
            trips_at_transfers[:, ii] = d_i
        
        d_un = (demand_matrix * (n_transfers > 2)).sum(dim=(1, 2))
        trips_at_transfers[:, 3] = d_un

        # calculate some more quantities of interest
        trip_times = state.transit_times.clone()
        trip_times[nopath] = 0
        demand_time = demand_matrix * trip_times
        total_dmd_time = demand_time.sum(dim=(1, 2))
        demand_transfers = demand_matrix * state.n_transfers
        total_transfers = demand_transfers.sum(dim=(1, 2))
        unserved_demand = (demand_matrix * nopath).sum(dim=(1, 2))
        total_demand = demand_matrix.sum(dim=(1,2))

        # compute total route times
        leg_times = get_route_leg_times(batch_routes, drive_times_matrix,
                                        self.mean_stop_time_s)
        total_route_time = leg_times.sum(dim=(1, 2))

        if self.symmetric_routes:
            transpose_dtm = drive_times_matrix.transpose(1, 2)
            return_leg_times = get_route_leg_times(batch_routes, transpose_dtm,
                                                   self.mean_stop_time_s)
            total_route_time += return_leg_times.sum(dim=(1, 2))

        output = CostHelperOutput(
            total_dmd_time, total_route_time, trips_at_transfers, 
            total_demand, unserved_demand, total_transfers, trip_times,
            n_disconnected_demand_edges, n_stops_oob, batch_routes
        )

        if return_per_route_riders:
            _, used_routes = \
                get_route_edge_matrix(batch_routes, drive_times_matrix,
                                      self.mean_stop_time_s, 
                                      self.symmetric_routes, 
                                      self.low_memory_mode, 
                                      return_used_routes=True)

            used_routes.unsqueeze_(-1)
            route_seqs = aggregate_edge_features(state.route_nexts, 
                                                 used_routes, 'concat')
            route_seqs.squeeze_(-1)
            per_route_riders = torch.zeros(batch_routes.shape[:2], device=dev)
            for bi in range(state.batch_size):
                for ri in range(batch_routes.shape[1]):
                    srcs, dsts, _ = torch.where(route_seqs[bi] == ri)
                    ri_demand = demand_matrix[bi, srcs, dsts].sum()
                    per_route_riders[bi, ri] = ri_demand

            output.per_route_riders = per_route_riders
        
        return output
        

class MyCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 min_route_len=2, max_route_len=None,
                 symmetric_routes=True, low_memory_mode=False,
                 demand_time_weight=0.5, route_time_weight=0.5, 
                 unserved_weight=5, variable_weights=False,
                 pp_fraction=0.33, op_fraction=0.33):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes, low_memory_mode)
        self.demand_time_weight = demand_time_weight
        self.route_time_weight = route_time_weight
        self.unserved_weight = unserved_weight
        self.variable_weights = variable_weights
        self.min_route_len = min_route_len
        self.max_route_len = max_route_len
        if self.variable_weights:
            # the fraction of variable weights sampled that are PP and OP
            self.pp_fraction = pp_fraction
            self.op_fraction = op_fraction
            assert pp_fraction + op_fraction <= 1, \
                "fractions of extreme samples must sum to <= 1"

    def sample_variable_weights(self, batch_size, device=None):
        if not self.variable_weights:
            dtw = torch.full((batch_size,), self.demand_time_weight, 
                             device=device)
            rtw = torch.full((batch_size,), self.route_time_weight, 
                              device=device)
        else:
            random_number = torch.rand(batch_size, device=device)
            # Initialized to zero, which is the right value for OP
            dtw = torch.zeros(batch_size, device=device)
            # Set demand time weight to 1 where we're using PP
            is_pp = random_number < self.pp_fraction
            dtw[is_pp] = 1.0
            # Set demand time weight to a random value in [0,1] where it's
             # neither OP nor PP
            extremes_fraction = self.pp_fraction + self.op_fraction
            is_intermediate = random_number >= extremes_fraction
            n_intermediate = is_intermediate.sum()
            dtw[is_intermediate] = torch.rand(n_intermediate, device=device)

            # reward time weight is 
            rtw = 1 - dtw
 
        return {
            'demand_time_weight': dtw,
            'route_time_weight': rtw
        }
    
    def get_weights(self, device=None):
        dtm = self.demand_time_weight
        if type(dtm) is not torch.Tensor:
            dtm = torch.tensor([dtm], device=device)
        rtm = self.route_time_weight
        if type(rtm) is not torch.Tensor:
            rtm = torch.tensor([rtm], device=device)
        
        return {
            'demand_time_weight': dtm,
            'route_time_weight': rtm
        }
    
    def set_weights(self, demand_time_weight=None, route_time_weight=None, 
                    unserved_weight=None):
        if demand_time_weight is not None:
            self.demand_time_weight = demand_time_weight
        if route_time_weight is not None:
            self.route_time_weight = route_time_weight
        if unserved_weight is not None:
            self.unserved_weight = unserved_weight

    def forward(self, state, unserved_weight=None, no_norm=False, 
                return_per_route_riders=False):
        cho = self._cost_helper(state, return_per_route_riders)
        cost_weights = state.cost_weights
        if 'demand_time_weight' in cost_weights:
            demand_time_weight = cost_weights['demand_time_weight']
        else:
            demand_time_weight = self.demand_time_weight
        if 'route_time_weight' in cost_weights:
            route_time_weight = cost_weights['route_time_weight']
        else:
            route_time_weight = self.route_time_weight
            
        if unserved_weight is None:
            unserved_weight = self.unserved_weight

        # if we have more weights than routes, truncate the weights
        if type(demand_time_weight) is torch.Tensor and \
           demand_time_weight.shape[0] > state.batch_size:
            demand_time_weight = demand_time_weight[:state.batch_size]
        if type(route_time_weight) is torch.Tensor and \
           route_time_weight.shape[0] > state.batch_size:
            route_time_weight = route_time_weight[:state.batch_size]
        if type(unserved_weight) is torch.Tensor and \
           unserved_weight.shape[0] > state.batch_size:
            unserved_weight = unserved_weight[:state.batch_size]

        # normalize all time values by the maximum drive time in the graph
        time_normalizer = state.drive_times.max(-1)[0].max(-1)[0]
        norm_dmd_time = cho.mean_demand_time / time_normalizer

        # normalize average route time
        routes = cho.batch_routes
        route_lens = (routes > -1).sum(-1)
        n_routes = (route_lens > 0).sum(-1)
        norm_route_time = \
            cho.total_route_time / (time_normalizer * n_routes + 1e-6)

        # fraction of demand not covered by routes, and fraction of routes
        n_demand_edges = (state.demand > 0).sum(dim=(1, 2))
        frac_uncovered = cho.n_disconnected_demand_edges / n_demand_edges
        if self.max_route_len is None:
            denom = n_routes * self.min_route_len
        else:
            denom = n_routes * self.max_route_len
        denom[denom == 0] = 1
        rld_frac = cho.n_stops_oob / denom
        constraints_violated = (rld_frac > 0) | (frac_uncovered > 0)
        cv_penalty = 0.1 * constraints_violated

        # average trip time, total route time, and trips-at-n-transfers
        if no_norm:
            cost = cho.mean_demand_time * demand_time_weight + \
                cho.total_route_time * route_time_weight
        else:
            cost = norm_dmd_time * demand_time_weight + \
                norm_route_time * route_time_weight            

        cost += (cv_penalty + frac_uncovered + rld_frac) * unserved_weight
        assert cost.isfinite().all()

        cho.cost = cost

        return cho


class NikolicCostModule(CostModule):
    def __init__(self, mean_stop_time_s=MEAN_STOP_TIME_S, 
                 avg_transfer_wait_time_s=AVG_TRANSFER_WAIT_TIME_S,
                 symmetric_routes=True, low_memory_mode=False,
                 unsatisfied_penalty_extra_s=UNSAT_PENALTY_EXTRA_S, 
                 ):
        super().__init__(mean_stop_time_s, avg_transfer_wait_time_s,
                         symmetric_routes, low_memory_mode)
        self.unsatisfied_penalty_extra_s = unsatisfied_penalty_extra_s
        self.min_route_len = 2
        self.max_route_len = None

    def forward(self, state):
        """
        symmetric_routes: if True, treat routes as going both ways along their
            stops.
        """
        cho = self._cost_helper(state)
        # Note that unlike Nikolic, we count trips that take >2 transfers as 
         # satisfied.
        tot_sat_demand = cho.total_demand - cho.unserved_demand
        w_2 = cho.total_demand_time / tot_sat_demand
        no_sat_dmd = torch.isclose(tot_sat_demand, 
                                   torch.zeros_like(tot_sat_demand))
        # if no demand is satisfied, set w_2 to the average time of all trips plus
        # the penalty
        w_2[no_sat_dmd] = cho.trip_times[no_sat_dmd].mean(dim=(-2,-1))
        w_2 += self.unsatisfied_penalty_extra_s

        cost = cho.total_demand_time + w_2 * cho.unserved_demand

        assert not ((cost == 0) & (cho.total_demand > 0)).any()

        log.debug("finished nikolic")
        assert cost.isfinite().all(), "invalid cost was computed!"

        cho.cost = cost
        
        return cho

    def get_weights(self, device=None):
        return {}


def check_for_duplicate_routes(routes_tensor):
    """check if any routes are duplicates of each other.

    In theory we want to avoid duplicate routes.  But no other work seems to 
    care.  Mumford doesn't check for them.b

    routes_tensor: a tensor of shape (batch_size, n_routes, route_len)"""
    routes_tensor = routes_tensor[:, None]
    same_stops = routes_tensor == routes_tensor.transpose(1, 2)
    routes_are_identical = same_stops.all(dim=-1)
    any_routes_are_identical = routes_are_identical.any(-1).any(-1)
    return any_routes_are_identical