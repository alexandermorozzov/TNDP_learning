import pytest
import pandas as pd
from pathlib import Path
import torch
from torch_geometric.data import Batch

from torch_utils import floyd_warshall
from simulation import DisaggregatedDemandSim
from simulation.transit_time_estimator import NikolicCostModule, MyCostModule, \
    RouteGenBatchState
from simulation.citygraph_dataset import CityGraphData, STOP_KEY


@pytest.mark.parametrize("env_dir", ['pt-simple', 'pt-multilines'])
# @pytest.mark.parametrize("env_dir", ['pt-multilines'])
def test_assign_trips_general(env_dir):
    full_env_dir = Path(__file__).parent / 'envs' / env_dir
    od_data = 'od.csv'
    sim = DisaggregatedDemandSim(od_data=od_data, basin_radius_m=500,
                                 delay_tolerance_s=1800, env_dir=full_env_dir)
    oddf = pd.read_csv(full_env_dir / od_data, dtype=str)
    # check that all trips in sim.tracked_trips have the correct dep and arr
    # check that all trips with a dep and arr in oddf are in sim.tracked_trips
    tracked_trips_by_id = {tt.id: tt for tt in sim.tracked_trips}
    for _, row in oddf.iterrows():
        if pd.isnull(row['true departure stop']) and \
           pd.isnull(row['true arrival stop']):
            # this passenger should not have taken transit
            assert row['ID'] not in tracked_trips_by_id
        else:
            # this passenger should begin and end their journeys at the
            # correct stops
            # TODO also test that begin and end times are correct
            true_dep_id = row['true departure stop']
            true_arr_id = row['true arrival stop']
            assert row['ID'] in tracked_trips_by_id
            planned_journey = tracked_trips_by_id[row['ID']].planned_journey
            assert planned_journey[0].board_stop_id == true_dep_id
            assert planned_journey[-1].disembark_stop_id == true_arr_id


def test_assign_trips_transferlimit():
    """Tests that reducing the max_transfers value reduces the allowed
    trip length.

    The environment contains four possible transit routes from the origin to
    the destination, requiring 1, 2, and 3 transfers.  Unrealistically, the
    more transfers a route has, the less time is the overall journey.  So
    without a limit, the journey with 3 transfers will be taken, with a limit
    of 3, the 2-transfer journey will be taken, etc.
    """
    full_env_dir = Path(__file__).parent / 'envs/switch-routes'
    od_data = 'od.csv'
    oddf = pd.read_csv(full_env_dir / od_data, dtype=str)
    # should be only one trip
    trip_id = oddf.iloc[0]['ID']
    sim = DisaggregatedDemandSim(od_data=od_data, basin_radius_m=500,
                                 delay_tolerance_s=1800, env_dir=full_env_dir,
                                 change_time_s=30, max_transfers=3)
    while sim.max_transfers > 0:
        assert len(sim.tracked_trips) == 1
        trip = sim.trips[trip_id]
        num_transfers = max(len(trip.planned_journey) - 1, 0)
        # check that the planned trip is below the limit
        assert num_transfers == sim.max_transfers
        # set the new max transfers to one less than the current transfers,
        # to force selection of a new route; then reassign
        sim.max_transfers = num_transfers - 1
        sim.assign_trips()
    assert len(sim.tracked_trips) == 0


# Nikolic test functions


class StaticTestCase:
    def __init__(self, street_edge_mat, demand_mat, times, routes, 
                 nikolic_kwargs, gt_nikolic_cost, mine_kwargs, gt_mine_cost,
                 gt_per_route_riders):
        self.street_edge_mat = street_edge_mat
        self.demand_mat = demand_mat
        self.times = times
        self.routes = routes
        self.nikolic_kwargs = nikolic_kwargs
        self.gt_nikolic_cost = gt_nikolic_cost
        self.mine_kwargs = mine_kwargs
        self.gt_mine_cost = gt_mine_cost
        self.gt_per_route_riders = gt_per_route_riders

    def __repr__(self):
        return f"NikolicTestCase(street_edge_mat={self.street_edge_mat}, " \
               f"demand_mat={self.demand_mat}, times={self.times}, " \
               f"routes={self.routes}, " \
               f"nikolic_kwargs={self.nikolic_kwargs}, " \
               f"gt_cost={self.gt_cost}, " \
               f"gt_route_costs={self.gt_route_costs}, " \
               f"per_route_riders={self.gt_per_route_riders}"


def check_static_test_case(test_case, low_memory_mode):
    nik_mod = NikolicCostModule(low_memory_mode=low_memory_mode,
                                **test_case.nikolic_kwargs)
    pseudo_data = CityGraphData()
    pseudo_data.drive_times = test_case.times
    pseudo_data.demand = test_case.demand_mat
    pseudo_data[STOP_KEY].x = torch.zeros((test_case.times.shape[1], 2))

    state = RouteGenBatchState(pseudo_data, nik_mod, len(test_case.routes[0]),
                               2, 100)                           
    state.add_new_routes([test_case.routes])

    results = nik_mod(state)
    assert torch.isclose(results.cost, test_case.gt_nikolic_cost).all()
    total_trips = results.trips_at_transfers.sum()
    assert total_trips == results.total_demand

    mine_mod = MyCostModule(low_memory_mode=low_memory_mode, 
                            **test_case.mine_kwargs)
    results = mine_mod(state)
    assert torch.isclose(results.cost, test_case.gt_mine_cost).all()
    total_trips = results.trips_at_transfers.sum()
    assert total_trips == results.total_demand

    per_route_riders = mine_mod(state, 
                                return_per_route_riders=True).per_route_riders
    assert torch.isclose(per_route_riders, test_case.gt_per_route_riders).all()


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_onedemand(low_memory_mode):
    check_static_test_case(get_onedemand_testcase(), low_memory_mode)


def get_onedemand_testcase(mean_stop_time_s=60):
    # a -> b
    nikolic_kwargs = {'mean_stop_time_s': mean_stop_time_s,
                      'symmetric_routes': False,}
    mine_kwargs = nikolic_kwargs.copy()
    route_time_weight = 0.5
    mine_kwargs['route_time_weight'] = route_time_weight
    demand_time_weight = 1 - route_time_weight
    mine_kwargs['demand_time_weight'] = demand_time_weight
    unserved_weight = 2
    mine_kwargs['unserved_weight'] = unserved_weight
    street_edge_mat = torch.tensor([[0, 1000], 
                                    [1000, 0]], dtype=torch.float32)
    # use unique powers of two for each element, so we can tell which demands
     # were included in any sum by the value of the sum
    demand_mat = torch.tensor([[0, 1], 
                               [0, 0]], dtype=torch.float32)
    # compute all shortest paths
    _, times = floyd_warshall(street_edge_mat)
    # route is a->b
    routes = [[0, 1]]
    demand_time = demand_mat.sum() * (1000 + mean_stop_time_s)
    nikolic_gt_cost = demand_time[None]

    mine_gt_cost = demand_time_weight * demand_time + \
        route_time_weight * (1000 + mean_stop_time_s)
    mine_gt_cost /= times.max()

    per_route_riders = torch.tensor([1], dtype=torch.float32)

    return StaticTestCase(street_edge_mat, demand_mat, times[0], routes, 
                          nikolic_kwargs, nikolic_gt_cost, mine_kwargs, 
                          mine_gt_cost, per_route_riders)


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_onedemand_symmetric(low_memory_mode):
    check_static_test_case(get_onedemand_symmetric_testcase(), low_memory_mode)


def get_onedemand_symmetric_testcase(mean_stop_time_s=60):
    # a -> b
    nikolic_kwargs = {'mean_stop_time_s': mean_stop_time_s,
                      'symmetric_routes': True,}
    mine_kwargs = nikolic_kwargs.copy()
    route_time_weight = 0.5
    mine_kwargs['route_time_weight'] = route_time_weight
    demand_time_weight = 1 - route_time_weight
    mine_kwargs['demand_time_weight'] = demand_time_weight
    unserved_weight = 2
    mine_kwargs['unserved_weight'] = unserved_weight
    street_edge_mat = torch.tensor([[0, 1000], 
                                    [1000, 0]], dtype=torch.float32)
    # use unique powers of two for each element, so we can tell which demands
     # were included in any sum by the value of the sum
    demand_mat = torch.tensor([[0, 0], 
                               [1, 0]], dtype=torch.float32)
    # compute all shortest paths
    _, times = floyd_warshall(street_edge_mat)
    # routes are a->b, b->c, and c->b->a
    routes = [[0, 1]]
    demand_time = demand_mat.sum() * (1000 + mean_stop_time_s)
    nikolic_gt_cost = demand_time[None]

    mine_gt_cost = demand_time_weight * demand_time + \
        2 * route_time_weight * (1000 + mean_stop_time_s)
    mine_gt_cost /= times.max()

    per_route_riders = torch.tensor([1], dtype=torch.float32)

    return StaticTestCase(street_edge_mat, demand_mat, times[0], routes, 
                          nikolic_kwargs, nikolic_gt_cost, mine_kwargs, 
                          mine_gt_cost, per_route_riders)


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_linear_3nodes(low_memory_mode):
    check_static_test_case(get_linear_3nodes_testcase(), low_memory_mode)


def get_linear_3nodes_testcase(mean_stop_time_s=60, transfer_time_s=300,
                               unsat_penalty=3000):
    common_kwargs = {'mean_stop_time_s': mean_stop_time_s,
                     'avg_transfer_wait_time_s': transfer_time_s,
                     'symmetric_routes': False,
                    }
    nikolic_kwargs = common_kwargs.copy()
    nikolic_kwargs['unsatisfied_penalty_extra_s'] = unsat_penalty

    mine_kwargs = common_kwargs.copy()
    route_time_weight = 0.5
    mine_kwargs['route_time_weight'] = route_time_weight
    demand_time_weight = 1 - route_time_weight
    mine_kwargs['demand_time_weight'] = demand_time_weight
    unserved_weight = 2
    mine_kwargs['unserved_weight'] = unserved_weight

    # a -> b -> c
    street_edge_mat = torch.tensor([[0, 1000, float('inf')], 
                                    [1000, 0, 1000], 
                                    [float('inf'), 1000, 0]], 
                                    dtype=torch.float32)
    # use unique powers of two for each element, so we can tell which demands
     # were included in any sum by the value of the sum
    demand_mat = torch.tensor([[0, 1, 2], 
                               [4, 0, 8], 
                               [16, 32, 0]], dtype=torch.float32)
    # compute all shortest paths
    _, times = floyd_warshall(street_edge_mat)
    # routes are a->b, b->c, and c->b->a
    routes = [[0, 1], [1, 2], [2, 1, 0]]

    street_edge_mat_plus_stoptime = street_edge_mat + mean_stop_time_s                                    
    _, times_for_cost = floyd_warshall(street_edge_mat_plus_stoptime)

    # check that the total cost is correct
    # unserved demand should be 0, demand * time should be:
    # 1000 + 2 * 2000 + 4 * 1000 + 8 * 1000 + 16 * 2000 + 32 * 1000 = 81000
    demand_time = (times_for_cost * demand_mat).sum()
    # 2 transfers, since one is required from 0 to 2
    n_transfers = demand_mat[0, 2]

    total_dmd_time = demand_time + transfer_time_s * n_transfers
    gt_nikolic_cost = total_dmd_time

    avg_dmd_time = total_dmd_time / demand_mat.sum()
    routes_time = (1000 + mean_stop_time_s) * 2 + (2000 + 2 * mean_stop_time_s)
    avg_route_time = routes_time / len(routes)
    gt_mine_cost = (avg_dmd_time * demand_time_weight + 
                    avg_route_time * route_time_weight) / times.max()
    
    per_route_riders = torch.tensor([3, 10, 16+32+4], dtype=torch.float32)

    return StaticTestCase(street_edge_mat, demand_mat, times[0], routes,
                          nikolic_kwargs, gt_nikolic_cost, mine_kwargs,
                          gt_mine_cost, per_route_riders)


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_3nodes_vshape(low_memory_mode):
    check_static_test_case(get_3nodes_vshape_testcase(), low_memory_mode)


def get_3nodes_vshape_testcase(mean_stop_time_s=60, transfer_time_s=300,
                               unsat_penalty=3000):
    # a <-> c
    #       ^
    #       v
    #       b
    common_kwargs = {'mean_stop_time_s': mean_stop_time_s,
                     'avg_transfer_wait_time_s': transfer_time_s,
                     'symmetric_routes': False}
    nikolic_kwargs = common_kwargs.copy()
    nikolic_kwargs['unsatisfied_penalty_extra_s'] = unsat_penalty

    mine_kwargs = common_kwargs.copy()
    route_time_weight = 0.5
    mine_kwargs['route_time_weight'] = route_time_weight
    demand_time_weight = 1 - route_time_weight
    mine_kwargs['demand_time_weight'] = demand_time_weight
    unserved_weight = 2
    mine_kwargs['unserved_weight'] = unserved_weight

    street_edge_mat = torch.tensor([[0, float('inf'), 1000], 
                                    [float('inf'), 0, 1000], 
                                    [1000, 1000, 0]], dtype=torch.float32)
    # use unique powers of two for each element, so we can tell which demands
     # were included in any sum by the value of the sum
    demand_mat = torch.tensor([ [0, 1, 2],
                                [4, 0, 8], 
                               [16, 32, 0]], dtype=torch.float32)
    # compute all shortest paths
    _, times = floyd_warshall(street_edge_mat)
    # routes are a->c, c->a, b->c, c->b
    routes = [[0, 2], [2, 0], [1, 2], [2, 1]]

    street_edge_mat_plus_stoptime = street_edge_mat + mean_stop_time_s                                    
    _, times_for_cost = floyd_warshall(street_edge_mat_plus_stoptime)

    # check that the total cost is correct
    # unserved demand should be 0
    demand_time = (times_for_cost * demand_mat).sum()
    # transfers are required from a to b and b to a
    n_transfers = demand_mat[0, 1] + demand_mat[1, 0]
    nikolic_gt_cost = demand_time + transfer_time_s * n_transfers

    route_time = (1000 + mean_stop_time_s)
    mine_gt_cost = nikolic_gt_cost / demand_mat.sum() * demand_time_weight
    mine_gt_cost += route_time_weight * route_time
    mine_gt_cost /= times.max()

    per_route_riders = torch.tensor([2+1, 16+4, 8+4, 32+1], 
                                    dtype=torch.float32)

    # all demand is satisfied
    return StaticTestCase(street_edge_mat, demand_mat, times[0], routes,
                           nikolic_kwargs, nikolic_gt_cost, mine_kwargs, 
                           mine_gt_cost, per_route_riders)


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_3nodes_loop_unsatdemand(low_memory_mode):
    check_static_test_case(get_3nodes_loop_unsatdemand_testcase(),
                           low_memory_mode)


def get_3nodes_loop_unsatdemand_testcase(mean_stop_time_s=60, 
                                         transfer_time_s=300,
                                         unsat_penalty=3000):
    # a <-> c
    #       ^
    #       v
    #       b
    common_kwargs = {'mean_stop_time_s': mean_stop_time_s,
                     'avg_transfer_wait_time_s': transfer_time_s,
                     'symmetric_routes': False}
    nikolic_kwargs = common_kwargs.copy()
    nikolic_kwargs['unsatisfied_penalty_extra_s'] = unsat_penalty

    mine_kwargs = common_kwargs.copy()
    route_time_weight = 0.5
    mine_kwargs['route_time_weight'] = route_time_weight
    demand_time_weight = 1 - route_time_weight
    mine_kwargs['demand_time_weight'] = demand_time_weight
    unserved_weight = 2
    mine_kwargs['unserved_weight'] = unserved_weight

    street_edge_mat = torch.tensor([[0, float('inf'), 1000], 
                                    [float('inf'), 0, 1000], 
                                    [1000, 1000, 0]], dtype=torch.float32)
    # use unique powers of two for each element, so we can tell which demands
     # were included in any sum by the value of the sum
    demand_mat = torch.tensor([ [0, 1, 2],
                                [4, 0, 8], 
                               [16, 32, 0]], dtype=torch.float32)
    # compute all shortest paths
    _, times = floyd_warshall(street_edge_mat)
    # routes are a->c, c->a, c->b
    routes = [[0, 2], [2, 0], [2, 1]]

    street_edge_mat_plus_stoptime = street_edge_mat + mean_stop_time_s                                    
    _, times_for_cost = floyd_warshall(street_edge_mat_plus_stoptime)

    # check that the total cost is correct
    # unserved demand should be 12
    sat_demand_mat = demand_mat.clone()
    sat_demand_mat[1, 0] = 0
    sat_demand_mat[1, 2] = 0
    demand_time = (times_for_cost * sat_demand_mat).sum()
    # transfers are accomplished from a to b, but not from b to a
    n_transfers = demand_mat[0, 1]
    travel_time = demand_time + transfer_time_s * n_transfers

    # include unsatisfied demand
    w_2 = travel_time / sat_demand_mat.sum() + unsat_penalty
    gt_nikolic_cost = travel_time + w_2 * (demand_mat[1, 0] + demand_mat[1, 2])

    gt_mine_cost = demand_time_weight * travel_time / sat_demand_mat.sum()
    gt_mine_cost += route_time_weight * (1000 + mean_stop_time_s)
    gt_mine_cost /= times.max()
    n_unserved = demand_mat[1, 0] + demand_mat[1, 2]
    # 2 node pairs out of nine are unbridged
    gt_mine_cost += unserved_weight * (0.1 + 2 / 6)

    per_route_riders = torch.tensor([2+1, 16+4, 32+1],
                                    dtype=torch.float32)

    return StaticTestCase(street_edge_mat, demand_mat, times[0], routes,
                          nikolic_kwargs, gt_nikolic_cost, mine_kwargs, 
                          gt_mine_cost, per_route_riders)


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_3nodes_loop_symmetric(low_memory_mode):
    check_static_test_case(get_3nodes_loop_symmetric_testcase(), 
                           low_memory_mode)


def get_3nodes_loop_symmetric_testcase(mean_stop_time_s=60, 
                                       transfer_time_s=300,
                                       unsat_penalty=3000):
    common_kwargs = {'mean_stop_time_s': mean_stop_time_s,
                     'avg_transfer_wait_time_s': transfer_time_s,
                     'symmetric_routes': True}
    nikolic_kwargs = common_kwargs.copy()
    nikolic_kwargs['unsatisfied_penalty_extra_s'] = unsat_penalty

    mine_kwargs = common_kwargs.copy()
    route_time_weight = 0.5
    mine_kwargs['route_time_weight'] = route_time_weight
    demand_time_weight = 1 - route_time_weight
    mine_kwargs['demand_time_weight'] = demand_time_weight
    unserved_weight = 2
    mine_kwargs['unserved_weight'] = unserved_weight

    street_edge_mat = torch.tensor([[0, float('inf'), 1000], 
                                    [float('inf'), 0, 1000], 
                                    [1000, 1000, 0]], dtype=torch.float32)
    # use unique powers of two for each element, so we can tell which demands
     # were included in any sum by the value of the sum
    demand_mat = torch.tensor([ [0, 1, 2],
                                [4, 0, 8], 
                               [16, 32, 0]], dtype=torch.float32)
    # compute all shortest paths
    _, times = floyd_warshall(street_edge_mat)
    # routes are a->c, c->a, c->b
    routes = [[0, 2], [2, 1]]

    street_edge_mat_plus_stoptime = street_edge_mat + mean_stop_time_s                                    
    _, times_for_cost = floyd_warshall(street_edge_mat_plus_stoptime)

    # check that the total cost is correct
    sat_demand_mat = demand_mat.clone()
    demand_time = (times_for_cost * sat_demand_mat).sum()
    # transfers are accomplished from a to b and b to a
    n_transfers = demand_mat[0, 1] + demand_mat[1, 0]
    travel_time = demand_time + transfer_time_s * n_transfers
    gt_nikolic_cost = travel_time

    gt_mine_cost = demand_time_weight * travel_time / sat_demand_mat.sum()
    gt_mine_cost += 2 * route_time_weight * (1000 + mean_stop_time_s)
    gt_mine_cost /= times.max()

    per_route_riders = torch.tensor([1+2+4+16, 8+4+32+1],
                                    dtype=torch.float32)
    
    return StaticTestCase(street_edge_mat, demand_mat, times[0], routes,
                          nikolic_kwargs, gt_nikolic_cost, mine_kwargs, 
                          gt_mine_cost, per_route_riders)


@pytest.mark.parametrize('low_memory_mode', [True, False])
def test_static_batched(low_memory_mode):
    # make sure we're using the same kwargs for all of these
    tc1 = get_linear_3nodes_testcase()
    test_cases = [tc1, get_onedemand_testcase(),
                  get_3nodes_vshape_testcase(),
                  get_3nodes_loop_unsatdemand_testcase()]
    # assemble the batch
    batch_routes = [tc.routes for tc in test_cases]
    max_n_nodes = max([tc.demand_mat.shape[0] for tc in test_cases])
    demand_tensor = torch.zeros((len(test_cases), max_n_nodes, max_n_nodes))
    times_tensor = demand_tensor.clone()
    gt_nik_costs = torch.zeros(len(test_cases), dtype=tc1.gt_nikolic_cost.dtype)
    gt_mine_costs = gt_nik_costs.clone()
    for bi, tc in enumerate(test_cases):
        n_nodes = tc.demand_mat.shape[0]
        demand_tensor[bi, :n_nodes, :n_nodes] = tc.demand_mat
        times_tensor[bi, :n_nodes, :n_nodes] = tc.times
        gt_nik_costs[bi] = tc.gt_nikolic_cost
        gt_mine_costs[bi] = tc.gt_mine_cost

    # run the batched version
    nik_mod = NikolicCostModule(low_memory_mode=low_memory_mode,
                                **tc1.nikolic_kwargs)
    # pseudo_data = DummyData(times_tensor, demand_tensor)
    pseudo_data = []
    n_nodess = [3, 2, 3, 3]
    for ii, (tc, n_nodes) in enumerate(zip(test_cases, n_nodess)):
        pd = CityGraphData()
        pd.drive_times = times_tensor[ii]
        pd.demand = demand_tensor[ii]
        pd[STOP_KEY].x = torch.zeros(n_nodes, 2)
        pseudo_data.append(pd)

    pseudo_data = Batch.from_data_list(pseudo_data)
    max_n_routes = max([len(tc.routes) for tc in test_cases])
    routes = [tc.routes for tc in test_cases]
    state = RouteGenBatchState(pseudo_data, nik_mod, max_n_routes, 2, 100)
    state.add_new_routes(routes)

    results = nik_mod(state)
    total_trips = results.trips_at_transfers.sum(dim=1)
    assert (total_trips == results.total_demand).all()
    assert torch.isclose(results.cost, gt_nik_costs).all()

    mine_mod = MyCostModule(low_memory_mode=low_memory_mode,
                            **tc1.mine_kwargs)
    results = mine_mod(state)
    total_trips = results.trips_at_transfers.sum(dim=1)
    assert (total_trips == results.total_demand).all()
    assert torch.isclose(results.cost, gt_mine_costs).all()


# from torch_geometric.loader import DataLoader

# def test_mandl():
#     four_routes = [
#         [0, 1, 2, 5, 7, 9, 10, 11],
#         [1, 4, 3, 5, 7, 9, 12, 10],
#         [8, 14, 6, 9, 7, 5, 3, 11],
#         [3, 1, 2, 5, 14, 6, 9, 13]
#     ]
#     four_routes_att = 10.51
#     four_routes_dx = [92.1, 7.19, 0.71, 0]
#     # load the mandl environment
#     # pass it to the Nikolic cost function
#     nik_kwargs = {
#         'mean_stop_time_s': 0,
#         'avg_transfer_wait_time_s': 300,
#         'symmetric_routes': True,
#         'unsatisfied_penalty_extra_s': 3000,
#     }
#     nik_mod = NikolicCostModule(**nik_kwargs)
#     city = CityGraphData.from_mumford_data(
#         'datasets/mumford_dataset/Instances', 'Mandl')
#     city = next(iter(DataLoader([city])))
#     result = nik_mod.get_metrics(four_routes, city)
#     att = result[0]
#     rtt = result[1]
#     dx = torch.tensor(result[2])
#     import pdb; pdb.set_trace()
#     assert torch.isclose(att, torch.tensor(four_routes_att))
#     assert torch.isclose(dx, four_routes_dx).all()


#     six_routes = [
#         [0, 1, 2, 5, 7, 9, 10, 12],
#         [0, 1, 4, 3, 5, 7, 9, 10],
#         [8, 14, 6, 9, 13, 12, 10, 11],
#         [0, 1, 2, 5, 14, 6, 9, 10],
#         [14, 7, 9, 10, 11, 3, 1, 0],
#         [8, 14, 5, 2, 1, 4, 3, 11]
#     ]
#     six_routes_att = 10.23
#     six_routes_dx = [95.63, 4.37, 0, 0]
#     seven_routes = [
#         [0, 1, 2, 5, 7, 9, 13, 12],
#         [0, 1, 4, 3, 5, 7, 9, 10],
#         [8, 14, 6, 9, 13, 12, 10, 11],
#         [0, 1, 2, 5, 14, 6, 9, 10],
#         [5, 7, 9, 10, 11, 3, 1, 0],
#         [8, 14, 7, 5, 2, 1, 3, 4],
#         [6, 14, 7, 5, 3, 11, 10, 12]        
#     ]
#     seven_routes_att = 10.15
#     seven_routes_dx = [98.52, 1.48, 0, 0]
#     eight_routes = [
#         [0, 1, 2, 5, 7, 9, 10, 12],
#         [2, 1, 4, 3, 5, 7, 14, 6],
#         [8, 14, 6, 9, 10, 11, 3, 5],
#         [0, 1, 2, 5, 14, 6, 9, 13],
#         [8, 14, 5, 2, 1, 3, 11],
#         [0, 1, 3, 11, 10, 12, 13, 9],
#         [1, 4, 3, 5, 7, 9, 10, 12],
#         [0, 1, 4, 3, 11, 10, 12, 13]
#     ]
#     eight_routes_att = 10.09
#     eight_routes_dx = [98.97, 1.03, 0, 0]

#     # import the Mandl environment