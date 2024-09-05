import random

import pytest
import hydra
from omegaconf import DictConfig
import torch

from simulation.citygraph_dataset import get_dataset_from_config, \
    DynamicCityGraphDataset, CityGraphData
from simulation.transit_time_estimator import RouteGenBatchState, \
    count_duplicate_stops, count_skipped_stops
import learning.utils as lrnu
import torch_utils as tu
import learning.heuristics as hs
from learning.initialization import john_init

INF = float('inf')

def get_cfg():
    try:
        hydra.initialize(config_path="../cfg", version_base=None)
    except ValueError:
        # hydra is already initialized
        pass
    cfg = hydra.compose(config_name='test_heuristics.yaml')
    return cfg


@pytest.fixture
def mandl_state():
    cfg = get_cfg()
    _, _, _, cost_obj, _ = \
        lrnu.process_standard_experiment_cfg(cfg)
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    evcfg = cfg.eval
    state = RouteGenBatchState(test_ds[0], cost_obj, evcfg.n_routes, 
                               evcfg.min_route_len, evcfg.max_route_len)
    return state

@pytest.fixture
def mandl_networks():
    # create some different networks
    network1 = [[1, 2, 3, 6, 8, 10, 14], 
                [9, 15, 7, 10, 11, 12], 
                [1, 2, 4, 12],
                [6, 8],
                [5, 4, 6, 15, 9],
                [4, 12, 11, 13, 14]]
    network2 = [[2, 3],
                [5, 4, 12, 11, 10],
                [6, 8, 15, 9],
                [7, 10, 14, 13],
                [1, 2, 4, 6],
                [6, 15, 7, 10, 13]]
    network3 = [[5, 4, 12, 11, 10],
                [4, 12, 11, 13, 14],
                [1, 2, 4, 6],
                [6, 8],
                [9, 15, 7, 10, 11, 12],
                [6, 8, 15, 9]]

    networks = tu.get_batch_tensor_from_routes([network1, network2, network3],
                                               max_route_len=8)
    networks -= 1
    networks.clamp_(min=-1)
    return networks


def test_get_add_node_options(mandl_state, mandl_networks):
    # unit tests for _get_add_node_options
    is_option = hs._get_add_node_options(mandl_state, mandl_networks, 'any')
    network1_options = torch.tensor([
        # 1, 2, 3, 6, 8, 10, 14
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 9, 15, 7, 10, 11, 12
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 1, 2, 4, 12
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],         
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 6, 8
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 5, 4, 6, 15, 9
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 4, 12, 11, 13, 14
        [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
    ], dtype=bool)

    network2_options =  torch.tensor([
        # 2, 3
        [[1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 5, 4, 12, 11, 10
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 6, 8, 15, 9
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 7, 10, 14, 13
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 1, 2, 4, 6
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 6, 15, 7, 10, 13
        [[0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
    ], dtype=bool)

    network3_options = torch.tensor([
         # 5, 4, 12, 11, 10
        [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 4, 12, 11, 13, 14
        [[0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 1, 2, 4, 6
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 6, 8
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 9, 15, 7, 10, 11, 12
        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
         # 6, 8, 15, 9
        [[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],],
    ], dtype=bool)

    networks_options = [network1_options, network2_options, network3_options]
    networks_options = torch.stack(networks_options)
    assert torch.all(is_option == networks_options)

    # test 'inside'
    is_option = hs._get_add_node_options(mandl_state, mandl_networks,
                                            'inside')
    gt_is_inside_option = networks_options.clone()
    gt_is_inside_option[:, :, 0] = False
    post_route_scatter_idcs = \
        hs._get_route_end_scatter_idcs(mandl_networks, networks_options) + 1
    gt_is_inside_option.scatter_(2, post_route_scatter_idcs, False)
    assert torch.all(is_option == gt_is_inside_option)

    # test 'terminal'
    is_option = hs._get_add_node_options(mandl_state, mandl_networks,
                                            'terminal')
    gt_is_term_option = torch.zeros_like(is_option)
    gt_is_term_option[:, :, 0] = networks_options[:, :, 0]
    end_is_option = torch.gather(networks_options, 2, post_route_scatter_idcs)
    gt_is_term_option.scatter_(2, post_route_scatter_idcs, end_is_option)
    assert torch.all(is_option == gt_is_term_option)


def test_get_delete_node_options(mandl_state, mandl_networks):
    is_option = hs._get_delete_node_options(mandl_state, mandl_networks, 'any')
    networks_options = torch.tensor([
        [[1, 0, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 1, 0, 0],       
         [1, 0, 0, 1, 0, 0, 0, 0],         
         [0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],         
         [1, 0, 0, 0, 1, 0, 0, 0],],
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0],       
         [1, 0, 0, 0, 0, 0, 0, 0],         
         [1, 0, 0, 1, 0, 0, 0, 0],         
         [0, 0, 0, 1, 0, 0, 0, 0], 
         [1, 0, 0, 0, 1, 0, 0, 0],],
        [[0, 0, 0, 0, 1, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 1, 0, 0],       
         [1, 1, 0, 1, 0, 0, 0, 0],],
    ], dtype=bool) 
    assert torch.all(is_option == networks_options)

    # test 'inside'
    is_option = hs._get_delete_node_options(mandl_state, mandl_networks, 
                                            'inside')
    gt_is_inside_option = networks_options.clone()
    gt_is_inside_option[:, :, 0] = False
    end_scatter_idcs = hs._get_route_end_scatter_idcs(mandl_networks, 
                                                      is_option)
    gt_is_inside_option.scatter_(2, end_scatter_idcs, False)
    assert torch.all(is_option == gt_is_inside_option)

    # test 'terminal'
    is_option = hs._get_delete_node_options(mandl_state, mandl_networks,
                                            'terminal')
    gt_is_term_option = torch.zeros_like(is_option)
    gt_is_term_option[:, :, 0] = networks_options[:, :, 0]
    end_is_option = torch.gather(networks_options, 2, end_scatter_idcs)
    gt_is_term_option.scatter_(2, end_scatter_idcs, end_is_option)
    assert torch.all(is_option == gt_is_term_option)


def test_get_served_pair_counts(mandl_state, mandl_networks):
    served_pair_counts = hs.get_served_pair_counts(mandl_state, mandl_networks)
    gt_spc = torch.tensor([
        [  # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
            [0, 2, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], # 0
            [2, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], # 1
            [1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0], # 2
            [1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 2, 1, 1, 1], # 3
            [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], # 4
            [1, 1, 1, 1, 1, 0, 0, 2, 1, 1, 0, 0, 0, 1, 1], # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1], # 6
            [1, 1, 1, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0], # 7
            [0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 2], # 8
            [1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1], # 9
            [0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 2, 1, 1, 1], # 10
            [1, 1, 0, 2, 0, 0, 1, 0, 1, 1, 2, 0, 1, 1, 1], # 11
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0], # 12
            [1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0], # 13
            [0, 0, 0, 1, 1, 1, 1, 0, 2, 1, 1, 1, 0, 0, 0], # 14
        ],
        [  # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
            [1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0], # 3
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], # 4
            [1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0, 2], # 5
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 2, 1, 1], # 6
            [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1], # 7
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], # 8
            [0, 0, 0, 1, 1, 1, 2, 0, 0, 0, 1, 1, 2, 1, 1], # 9
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0], # 10
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # 11
            [0, 0, 0, 0, 0, 1, 2, 0, 0, 2, 0, 0, 0, 1, 1], # 12
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0], # 13
            [0, 0, 0, 0, 0, 2, 1, 1, 1, 1, 0, 0, 1, 0, 0], # 14
        ],
        [  # 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
            [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 0
            [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 1
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # 2
            [1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 2, 2, 1, 1, 0], # 3
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], # 4
            [1, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 1], # 5
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1], # 6
            [0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1], # 7
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 2], # 8
            [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 2, 2, 0, 0, 1], # 9
            [0, 0, 0, 2, 1, 0, 1, 0, 1, 2, 0, 3, 1, 1, 1], # 10
            [0, 0, 0, 2, 1, 0, 1, 0, 1, 2, 3, 0, 1, 1, 1], # 11
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0], # 12
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0], # 13
            [0, 0, 0, 0, 0, 1, 1, 1, 2, 1, 1, 1, 0, 0, 0], # 14
        ],
    ])
    assert (gt_spc == served_pair_counts).all()


@pytest.fixture
def random_cities_and_networks():
    torch.manual_seed(1)
    random.seed(1)
    generator = DynamicCityGraphDataset(15, 50)
    # n_nodes, n_routes, min_stops, max_stops
    city_params = [(15, 5, 2, 8), 
                   (20, 8, 3, 10), 
                   (30, 10, 4, 15),
                   (50, 20, 5, 20)] * 5
    cities = [generator.generate_graph(n_nodes=nn[0], max_n_nodes=nn[0]) 
              for nn in city_params]
    
    # make RouteGenBatchState objects from the cities
    cfg = get_cfg()
    _, _, _, cost_obj, _ = \
        lrnu.process_standard_experiment_cfg(cfg)
    cities = [RouteGenBatchState(city, cost_obj, *nn[1:]) 
              for (city, nn) in zip(cities, city_params)]

    # generate networks
    networks = [john_init(city, torch.rand(1)) for city in cities]
    return cities, networks


def _test_validity_helper(states, networks, func_to_test, 
                          n_routes_should_change=1, n_repetitions=10):
    # generate some random graphs and networks
    cfg = get_cfg()
    _, _, _, cost_obj, _ = \
        lrnu.process_standard_experiment_cfg(cfg)
    
    for state, network in zip(states, networks):
        shortest_paths, _ = tu.reconstruct_all_paths(state.graph_data.nexts)
        sp_lens = (shortest_paths > -1).sum(dim=-1)
        assert state.street_adj.isinf().any()

        for it in range(n_repetitions):
            # apply add_terminal
            old_network = network
            network = func_to_test(state, old_network)
            # check that all nodes are covered
            mask = tu.get_nodes_on_routes_mask(state.max_n_nodes, network)
            nodes_are_covered = mask[..., :-1].any(dim=1)
            assert nodes_are_covered.all()

            # check that all routes are valid lengths
            state.replace_routes(network)
            cost_result = cost_obj(state)
            assert (cost_result.n_stops_oob == 0).all()
            
            # check that there are no loops
            n_dups = count_duplicate_stops(state.max_n_nodes, network)
            assert (n_dups == 0).all()

            # check that routes don't skip any nodes
            total_n_skips = count_skipped_stops(network, sp_lens)
            assert (total_n_skips == 0).all()

            # check that no more than one route is changed
            routes_are_changed = (old_network != network).any(dim=-1)
            n_changed_routes = routes_are_changed.sum().item()
            # sometimes no change is possible, so it might be 0
            assert n_changed_routes in [0, n_routes_should_change]


def test_add_terminal(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.add_terminal)


def test_delete_terminal(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.delete_terminal)


def test_cost_based_trim(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.cost_based_trim)


@pytest.fixture
def linear_city_state():
    # first, a small city graph with five nodes in a line
    adj_mat = torch.tensor([
        [0, 10, INF, INF, INF],
        [10, 0, 10, INF, INF],
        [INF, 10, 0, 10, INF],
        [INF, INF, 10, 0, 10],
        [INF, INF, INF, 10, 0]
    ])
    demand = torch.tensor([
        [0, 0,  0, 0, 0],
        [0, 0, 10, 10, 0],
        [0, 10, 0, 10, 0],
        [0, 10, 10, 0, 0],
        [0, 0,  0, 0, 0],
    ])

    node_locs = torch.tensor([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0]
    ])

    graph_data = CityGraphData.from_tensors(node_locs, adj_mat, demand)
    cfg = get_cfg()
    _, _, _, cost_obj, _ = \
        lrnu.process_standard_experiment_cfg(cfg)
    max_len=5
    state = RouteGenBatchState(graph_data, cost_obj, 0, 2, max_len)
    return state


@pytest.fixture
def branching_city_state():
    # a city with three branches
    #         2     
    #       /   \
    # 0 -- 1 --- 3 -- 5
    #       \   /
    #         4 
    # all edges are the same length, except the one across the "middle"
    adj_mat = torch.tensor([
        [0,   10,  INF, INF, INF, INF],
        [10,   0,  10,   15, 10,  INF],
        [INF, 10,   0,   10, INF, INF],
        [INF, 15,  10,    0, 10,   10],
        [INF, 10, INF,   10, 0,   INF],
        [INF, INF, INF,  10, INF,   0],
    ])
    demand = torch.tensor([
        [0, 10,  0,  0,  0, 10],
        [10, 0, 10, 10, 10,  0],
        [0, 10,  0, 10, 10,  0],
        [0, 10, 10,  0, 10, 10],
        [0, 10, 10, 10,  0,  0],
        [10, 0,  0, 10,  0,  0],
    ])

    node_locs = torch.tensor([
        [0, 0],
        [1, 0],
        [2, 1],  # branch 1
        [3, 0],  # branch 2
        [2, -1], # branch 3
        [4, 0],
    ])

    graph_data = CityGraphData.from_tensors(node_locs, adj_mat, demand)
    cfg = get_cfg()
    _, _, _, cost_obj, _ = \
        lrnu.process_standard_experiment_cfg(cfg)
    max_len=5
    state = RouteGenBatchState(graph_data, cost_obj, 0, 2, max_len)
    return state


def test_cost_based_trim_probs_linear(linear_city_state):
    # test the probabilities

    # one route on the first two nodes, one on the last two, and one on the 
     # middle three
    networks = [
        [[0, 1],
         [1, 2, 3],
         [3, 4]],
         # then the same but the middle one covers the last four routes
        [[0, 1],
         [1, 2, 3, 4],
         [3, 4]],
        ]
    max_len = linear_city_state.max_route_len[0]
    networks = tu.get_batch_tensor_from_routes(networks, max_route_len=max_len)

    probs = hs._cost_based_trim_probs(linear_city_state, networks)
    # only middle route in each network is trimmable
    # probabilities should be zero for the first and last routes
    assert (probs[0, 0] == 0).all()
    assert (probs[0, 2] == 0).all()
    # probabilities should be equal for the two terminals on route 1
    gt_route1probs = torch.tensor([0.5, 0, 0.5, 0, 0], dtype=torch.float32)
    assert torch.isclose(probs[0, 1], gt_route1probs).all()

    # check second network
    # probabilities should be zero for the first and last routes
    assert (probs[1, 0] == 0).all()
    assert (probs[1, 2] == 0).all()
    # probabilities should be 0.5 for the start term, 1e7 on the end term
    gt_route2probs = torch.tensor([0.5, 0, 0, 10 / hs.EPSILON, 0], 
                                  dtype=torch.float32)
    assert torch.isclose(probs[1, 1], gt_route2probs).all()

    # remove the demand between 1 and 3
    linear_city_state.demand[0, 1, 3] = 0
    linear_city_state.demand[0, 3, 1] = 0
    # now the probabilities should be 1 instead of 0.5 for the two terminals
    probs = hs._cost_based_trim_probs(linear_city_state, networks)
    gt_route1probs = torch.tensor([1, 0, 1, 0, 0], dtype=torch.float32)
    assert torch.isclose(probs[0, 1], gt_route1probs).all()

    gt_route2probs = torch.tensor([1, 0, 0, 10 / hs.EPSILON, 0], 
                                   dtype=torch.float32)
    assert torch.isclose(probs[1, 1], gt_route2probs).all()

    linear_city_state.demand[0, 1, 2] = 30
    linear_city_state.demand[0, 2, 1] = 30
    probs = hs._cost_based_trim_probs(linear_city_state, networks)
    gt_route1probs = torch.tensor([1 / 3, 0, 1, 0, 0], dtype=torch.float32)
    assert torch.isclose(probs[0, 1], gt_route1probs).all()

    gt_route2probs = torch.tensor([1 / 3, 0, 0, 10 / hs.EPSILON, 0], 
                                  dtype=torch.float32)
    assert torch.isclose(probs[1, 1], gt_route2probs).all()

    linear_city_state.demand[0, 1, 3] = 6
    linear_city_state.demand[0, 3, 1] = 6
    probs = hs._cost_based_trim_probs(linear_city_state, networks)
    gt_route1probs = torch.tensor([1 / 3.6, 0, 1 / 1.6, 0, 0], 
                                  dtype=torch.float32)
    assert torch.isclose(probs[0, 1], gt_route1probs).all()
    gt_route2probs = torch.tensor([1 / 3.6, 0, 0, 10 / hs.EPSILON, 0], 
                                  dtype=torch.float32)
    assert torch.isclose(probs[1, 1], gt_route2probs).all()

    # adjust the street adj times
    linear_city_state.street_adj[0, 2, 3] = 2
    linear_city_state.street_adj[0, 3, 2] = 2
    probs = hs._cost_based_trim_probs(linear_city_state, networks)
    gt_route1probs = torch.tensor([1 / 3.6, 0, 1 / 8, 0, 0], 
                                  dtype=torch.float32)
    assert torch.isclose(probs[0, 1], gt_route1probs).all()
    gt_route2probs = torch.tensor([1 / 3.6, 0, 0, 10 / hs.EPSILON, 0], 
                                  dtype=torch.float32)
    assert torch.isclose(probs[1, 1], gt_route2probs).all()

    # now try two routes that can be shortened
    # put the street adjacencies back the way they were
    linear_city_state.street_adj[0, 2, 3] = 10
    linear_city_state.street_adj[0, 3, 2] = 10

    networks = [
        [[0, 1, 2],
         [1, 2, 3],
         [3, 4]],
        [[0, 1, 2],
         [1,2,3,4],
         [3, 4]],
    ]
    networks = tu.get_batch_tensor_from_routes(networks, max_route_len=max_len)

    # repeat ten times to hopefully cover both branches
    for _ in range(10):
        probs = hs._cost_based_trim_probs(linear_city_state, networks)
        # first network
        route0_probs_are_zero = (probs[0, 0] == 0).all()
        route1_probs_are_zero = (probs[0, 1] == 0).all()
        # one should be all 0s, the other not
        gt_route1probs = torch.tensor([10 / 6, 0, 1 / 1.6, 0, 0], 
                                      dtype=torch.float32)
        gt_route0probs = torch.tensor([0, 0, 10 / hs.EPSILON, 0, 0], 
                                      dtype=torch.float32)
        assert route0_probs_are_zero ^ route1_probs_are_zero
        if route0_probs_are_zero:
            assert torch.isclose(probs[0, 1], gt_route1probs).all()
        elif route1_probs_are_zero:
            assert torch.isclose(probs[0, 0], gt_route0probs).all()

        # second network
        route0_probs_are_zero = (probs[1, 0] == 0).all()
        route1_probs_are_zero = (probs[1, 1] == 0).all()
        assert route0_probs_are_zero != route1_probs_are_zero
        gt_route1probs = torch.tensor([10 / 6, 0, 0, 10 / hs.EPSILON, 0], 
                                       dtype=torch.float32)
        if route0_probs_are_zero:
            assert torch.isclose(probs[1, 1], gt_route1probs).all()
        elif route1_probs_are_zero:
            # gt probs are the same as in the first network
            assert torch.isclose(probs[1, 0], gt_route0probs).all()


def test_cost_based_trim_probs_branching(branching_city_state):
    networks = [
        [[0, 1],
         [1, 3, 5],
         [4, 3, 5]],
        [[0, 1, 2],
         [1, 2, 3],
         [1, 4, 3]],
    ]
    max_len = branching_city_state.max_route_len[0]
    networks = tu.get_batch_tensor_from_routes(networks, max_route_len=max_len)
    
    gt = torch.zeros((2, 3, max_len), dtype=torch.float32)
    # first network's second route can be shortened at either end
    gt[0, 1, 0] = 1.5
    gt[0, 1, 2] = 10 / hs.EPSILON
    # first network's third route can be shortened only at the end
    gt[0, 2, 2] = 10 / hs.EPSILON

    # second network's first route can be shortened only at the end
    gt[1, 0, 2] = 10 / hs.EPSILON
    # second network's second route can be shortened at either end
    gt[1, 1, 0] = 10 / hs.EPSILON
    gt[1, 1, 2] = 1.0
    # second network's third route can be shortened at either end
    gt[1, 2, 0] = 1.0
    gt[1, 2, 2] = 1.0

    for _ in range(20):
        probs = hs._cost_based_trim_probs(branching_city_state, networks)
        are_chosen = probs.any(dim=-1)
        assert (are_chosen.sum(dim=1) == 1).all()
        chosen_gt = gt * are_chosen[..., None]
        assert torch.isclose(probs, chosen_gt).all()

    # these are all too short to be deleted
    networks = [
        [[0, 1],
         [1, 2],
         [1, 3],
         [1, 4],
         [3, 5]],
        [[0, 1],
         [1, 2],
         [1, 3],
         [4, 3],
         [3, 5]],
    ]
    networks = tu.get_batch_tensor_from_routes(networks, max_route_len=max_len)
    for _ in range(20):
        probs = hs._cost_based_trim_probs(branching_city_state, networks)
        assert (probs == 0).all()


def test_cost_based_grow(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.cost_based_grow)


def test_cost_based_grow_probs_linear(linear_city_state):
    networks = [
        [[0, 1],
         [1, 2, 3],
         [3, 4]],
        [[0, 1],
         [1, 2, 3, 4],
         [3, 4]],
    ]
    max_len = linear_city_state.max_route_len[0]
    networks = tu.get_batch_tensor_from_routes(networks, max_route_len=max_len)

    n_nodes = linear_city_state.max_n_nodes
    gt = torch.zeros((2, 3, max_len+1, n_nodes), dtype=torch.float32)

    gt[0, 0, 2, 2] = 1

    gt[0, 1, 0, 0] = 1
    gt[0, 1, 3, 4] = 1

    gt[0, 2, 0, 2] = 1

    gt[1, 0, 2, 2] = 1
    gt[1, 1, 0, 0] = 1
    gt[1, 2, 0, 2] = 1

    for _ in range(20):
        probs = hs._cost_based_grow_probs(linear_city_state, networks)
        are_chosen = probs.flatten(start_dim=2).any(dim=-1)
        assert (are_chosen.sum(dim=1) == 1).all()

        chosen_gt = gt * are_chosen[..., None, None]
        assert torch.isclose(probs, chosen_gt).all()        

    # a test case where some non-directly-satisfied demand gets covered
    linear_city_state.demand[0, 0, 2] = 7
    linear_city_state.demand[0, 2, 0] = 7
    gt[...] = 0
    # first route can only grow to node 2, but new demand is satisfied!
    gt[0, 0, 2, 2] = 7/10
    gt[0, 1, 0, 0] = 7/10
    gt[0, 2, 0, 2] = 1

    gt[1, 0, 2, 2] = 7/10
    gt[1, 1, 0, 0] = 7/10
    gt[1, 2, 0, 2] = 1

    for _ in range(20):
        # repeat this multiple times to check each route's being chosen
        probs = hs._cost_based_grow_probs(linear_city_state, networks)
        are_chosen = probs.flatten(start_dim=2).any(dim=-1)
        assert (are_chosen.sum(dim=1) == 1).all()

        chosen_gt = gt * are_chosen[..., None, None]
        assert torch.isclose(probs, chosen_gt).all()        


def test_cost_based_grow_probs_branching(branching_city_state):
    # first, a network with three routes, one on each branch
    networks = [
        [[0, 1],
         [2, 1, 4],
         [3, 5]],

        [[0, 1, 2],
         [4, 3, 5],
         [1, 4]],
    ]
    max_len = branching_city_state.max_route_len[0]
    networks = tu.get_batch_tensor_from_routes(networks, max_route_len=max_len)
    
    n_nodes = branching_city_state.max_n_nodes
    gt = torch.zeros((2, 3, max_len+1, n_nodes), dtype=torch.float32)
    # 1st route in 1st network can grow to 2, 3, or 4, but only 3 has demand
    gt[0, 0, 2, 3] = 2/3
    # 2nd route in 1st network can grow to 0 or 3, with 3x10 demand each
    gt[0, 1, 0, 3] = 3
    gt[0, 1, 3, 3] = 3
    # 3rd route in 1st network can grow at its start
    gt[0, 2, 0, 1] = 2/3
    gt[0, 2, 0, 2] = 1
    gt[0, 2, 0, 4] = 1

    # 1st route in 2nd network could extend to 3
    gt[1, 0, 3, 3] = 2
    # 2nd route in 2nd network could extend to 1
    gt[1, 1, 0, 1] = 1
    # 3rd route in 2nd network could extend to 0 or 2 at the start
    # 0 has no demand
    gt[1, 2, 0, 0] = 0
    gt[1, 2, 0, 2] = 1
    gt[1, 2, 0, 3] = 2/3
    # and to 3 at the end
    gt[1, 2, 2, 3] = 1

    for _ in range(20):
        probs = hs._cost_based_grow_probs(branching_city_state, networks)
        are_chosen = probs.flatten(start_dim=2).any(dim=-1)
        assert (are_chosen.sum(dim=1) == 1).all()

        chosen_gt = gt * are_chosen[..., None, None]
        assert torch.isclose(probs, chosen_gt).all()

    # check that when no new demand is ever satisfied, all probs are equal
    branching_city_state.demand[...] = 0
    gt = hs._get_add_node_options(branching_city_state, networks, 'terminal')
    # remove dummy node options
    gt = gt[..., :-1].float()

    for _ in range(20):
        probs = hs._cost_based_grow_probs(branching_city_state, networks)
        are_chosen = probs.flatten(start_dim=2).any(dim=-1)
        assert (are_chosen.sum(dim=1) == 1).all()

        chosen_gt = gt * are_chosen[..., None, None]
        assert torch.isclose(probs, chosen_gt).all()



def test_add_inside(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.add_inside)


def test_delete_inside(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.delete_inside)


def test_invert_nodes(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.invert_nodes)


def test_exchange_routes(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.exchange_routes, 
                          n_routes_should_change=2)


def test_replace_node(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.replace_node)


def test_donate_node(random_cities_and_networks):
    states, networks = random_cities_and_networks
    _test_validity_helper(states, networks, hs.donate_node,
                          n_routes_should_change=2)