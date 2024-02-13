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

import argparse
from itertools import cycle

import omegaconf
import torch
import networkx as nx
import torch_geometric.utils as pygu
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import pickle

from simulation.citygraph_dataset import get_dataset_from_config, \
    STOP_KEY, STREET_KEY


import matplotlib
matplotlib.rcParams.update({'font.size': 20, 'pdf.fonttype': 42,
                            'ps.fonttype': 42})


def plot_routes(data, routes, show_demand=False, node_size=100, edge_size=10):
    city_pos = data[STOP_KEY].pos

    if show_demand:
        demand = data.demand
        nx_dmd_graph = nx.from_numpy_array(demand.numpy(), 
                                        create_using=nx.DiGraph)
        de_widths = torch.tensor([dd['weight'] for _, _, dd in 
                                    nx_dmd_graph.edges(data=True)])
        de_widths *= 2 / de_widths.max()
        nx.draw_networkx_edges(nx_dmd_graph, edge_color="red", 
                               pos=city_pos.numpy(), style="dashed", 
                               width=de_widths)

    edges = data[STREET_KEY].edge_index
    graph = Data(pos=city_pos, edge_index=edges)
    nx_graph = pygu.to_networkx(graph)

    ax = plt.gca()

    # draw the street graph
    if routes is None:
        nx.draw(nx_graph, pos=city_pos.numpy(), arrows=False, 
                node_size=node_size, width=edge_size, ax=ax)

    # draw the routes
    else:
        nx.draw_networkx_nodes(nx_graph, pos=city_pos.numpy(), 
                               node_size=node_size)
                            
        colours = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        min_width = edge_size * 0.2
        max_width = edge_size
        width_step = (max_width - min_width) / (len(routes) - 1)
        width = max_width
        for route_id, route in enumerate(routes):
            route_graph = nx.DiGraph()
            poss = {}
            route_len = (route >= 0).sum()
            route = route[:route_len]
            for si, ii in enumerate(route):
                route_graph.add_node(si)
                if si > 0:
                    route_graph.add_edge(si - 1, si)
                poss[si] = city_pos[ii].numpy()
            
            colour = next(colours)
            edge_col = nx.draw_networkx_edges(route_graph, pos=poss, 
                                              edge_color=colour, width=width, 
                                              arrowsize=width*2,
                                              node_size=0, ax=ax)
            width -= width_step
            edge_col[0].set_label(str(route_id))

    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    # plt.xlabel("meters")
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset config path")
    parser.add_argument("--routes", help=".pkl file with generated routes")
    parser.add_argument("--demand", action="store_true", 
                        help="show the demand on the graph")
    parser.add_argument("--ns", "--node_size", type=float,
                        help="size of displayed nodes")
    parser.add_argument("--es", "--edge_size", type=float,
                        help="thickness of displayed edges")
    args = parser.parse_args()

    ds_cfg = omegaconf.OmegaConf.load(args.dataset)
    if 'dataset' in ds_cfg:
        # the dataset is nested in the eval config, so pull it out
        ds_cfg = ds_cfg.dataset

    dataset = get_dataset_from_config(ds_cfg)

    if args.routes is None:
        # just plot the street graph
        for data in dataset:
            plot_routes(data, None, show_demand=args.demand, node_size=args.ns,
                        edge_size=args.es)

    else:
        # load the routes
        with open(args.routes, "rb") as ff:
            unloaded = pickle.load(ff)

            for routes, data in zip(unloaded, dataset):
                plot_routes(data, routes, show_demand=args.demand, 
                            node_size=args.ns, edge_size=args.es)


if __name__ == "__main__":
    main()