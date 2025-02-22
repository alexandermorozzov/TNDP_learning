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
from collections import defaultdict

import yaml
import omegaconf
import numpy as np
import torch
import networkx as nx
import torch_geometric.utils as pygu
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import pickle
import colorsys
import pyproj

import plotting_utils as pu
from simulation.citygraph_dataset import get_dataset_from_config, \
    STOP_KEY, STREET_KEY
import torch_utils as tu

import matplotlib
# set up matplotlib and seaborn
# matplotlib.rcParams['text.usetex'] = True
# force type-1 fonts
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True

matplotlib.rcParams['font.size'] = 20
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'serif'

matplotlib.rcParams['figure.constrained_layout.use'] = True


def map_latlons_to_nodes(latlon_dict, node_coords, mapframe="EPSG:3347"):
    """
    Maps latlons to the closest node in node_latlons
    """
    transformer = pyproj.Transformer.from_crs("EPSG:4326", mapframe)
    xys = {kk: transformer.transform(*vv) for kk, vv in 
           latlon_dict.items()}
    nodes_dict = {}
    for kk, (xx, yy) in xys.items():
        idx = np.argmin(np.linalg.norm(node_coords - [xx, yy], axis=1))
        nodes_dict[kk] = idx.item()
    return nodes_dict


def standardize_dir(direction):
    """Normalize the vector and make it point to the right and up"""
    norm = np.linalg.norm(direction)
    if norm == 0:
        return direction
    direction /= norm
    if direction[0] < 0:
        # flip it to point to the right
        direction *= -1
    if direction[0] == 0 and direction[1] < 0:
        # it's exactly vertical and points down, so flip it to point up
        direction *= -1
    return direction


def get_line_eqn(start, end):
    """Get the equation of the line between start and end"""
    # y = mx + b
    assert (end[0] != start[0]).all(), "points are the same!"
    mm = (end[1] - start[1]) / (end[0] - start[0])
    bb = start[1] - mm * start[0]
    return mm, bb


def get_offset_corner(point1, point2, ext_point, edge_offset, corner_at_1):
    """Get the offset for the corner point
    
        corner_at_1: True if point1 is the corner, False if point2 is the corner
    """
    if ext_point is None or (edge_offset == 0).all():
        if corner_at_1:
            return point1 + edge_offset
        else:
            return point2 + edge_offset
    
    edge_dir = point2 - point1
    perp_dir = standardize_dir(np.array([edge_dir[1], -edge_dir[0]]))
    if corner_at_1:
        corner_point = point1
        ext_edge_dir = point1 - ext_point
    else:
        corner_point = point2
        ext_edge_dir = ext_point - point2
    ext_perp_dir = standardize_dir(
        np.array([ext_edge_dir[1], -ext_edge_dir[0]]))
    corner_dir = (perp_dir + ext_perp_dir) / 2
    
    # dot product might be slightly outside [-1, 1] due to floating point error
    corner_perp_cos = max(-1.0, min(1.0, np.dot(corner_dir, perp_dir)))
    offset_corner_angle = np.arccos(corner_perp_cos)
    # ensure corner dir point to the same side of the edge as the offset
    if offset_corner_angle > np.pi / 2:
        # flip it so it's on the same side as the offset
        corner_dir *= -1

    # find intersection of corner_dir and offset edge direction
    offset_edge_points = (point1 + edge_offset, point2 + edge_offset)
    # do this to avoid floating-point wierdness when corner_dir is very small
     # compared to corner_point
    corner_dir *= np.linalg.norm(corner_point)
    corner_dir_points = (corner_point, corner_point + corner_dir)
    # find where the two lines intersect
    m1, b1 = get_line_eqn(*offset_edge_points)
    m2, b2 = get_line_eqn(*corner_dir_points)

    assert m1 != m2, "Lines are parallel"
    x_intersect = (b2 - b1) / (m1 - m2)
    y_intersect = m1 * x_intersect + b1
    return np.array([x_intersect, y_intersect])


def plot_routes(args, data, routes=None):
    city_pos = data[STOP_KEY].pos
    np_city_pos = city_pos.numpy()

    if args.demand:
        demand = data.demand
        nx_dmd_graph = nx.from_numpy_array(demand.numpy(), 
                                        create_using=nx.DiGraph)
        de_widths = torch.tensor([dd['weight'] for _, _, dd in 
                                    nx_dmd_graph.edges(data=True)])
        de_widths *= 2 / de_widths.max()
        nx.draw_networkx_edges(nx_dmd_graph, edge_color="red", 
                               pos=np_city_pos, style="dashed", 
                               width=de_widths)

    edges = data[STREET_KEY].edge_index
    graph = Data(pos=city_pos, edge_index=edges)
    nx_graph = pygu.to_networkx(graph)

    ax = plt.gca()

    # include special nodes, if there are any
    if args.special_coords:
        with open(args.special_coords, 'r') as ff:
            special_coords = yaml.safe_load(ff)
            special_latlons = {kk: vv['coordinates'] 
                               for kk, vv in special_coords.items()}
        special_nodes = map_latlons_to_nodes(special_latlons, np_city_pos)

        plot_locs = np.array([np_city_pos[nn] for nn in special_nodes.values()])
        markersize = 100
        # use a big z-order so that these are on top
        big_zorder = 100 if routes is None else 2 * len(routes)

        ax.scatter(plot_locs[:, 0], plot_locs[:, 1], color='black', 
                   s=markersize, zorder=big_zorder)
        if args.label_special:
            # get vertical range of the nodes
            x_min, y_min = np_city_pos.min(0)
            x_max, y_max = np_city_pos.max(0)
            text_offsets = np.array([vv['text_offset']
                                    for vv in special_coords.values()])
            text_offsets[:, 0] *= x_max - x_min
            text_offsets[:, 1] *= y_max - y_min
            text_locs = plot_locs + text_offsets
            for node_name, loc in zip(special_nodes.keys(), text_locs):
                ax.text(loc[0], loc[1], node_name, fontsize=15, 
                        zorder=big_zorder, va='center', ha='center')

        # finally, count how many routes stop at each special point
        if routes is not None:
            counts = {}
            for node_name, idx in special_nodes.items():
                n_stops = (routes == idx).sum().item()
                counts[node_name] = n_stops
            sorted_count_names = sorted(counts.keys())
            # print(sorted_count_names)
            counts = [counts[nn] for nn in sorted_count_names]
            csv_str = ",".join([str(cc) for cc in counts])

    # draw the street graph
    if routes is None:
        nx.draw(nx_graph, pos=np_city_pos, arrows=False, 
                node_size=args.ns, width=args.es, ax=ax)
        
    # draw the routes
    else:
        nx.draw_networkx_nodes(nx_graph, pos=np_city_pos, node_size=args.ns)
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color']
        count = len(routes)
        if len(routes) > len(colours):
            # not enough colours, so generate a set of colours
            hsv_colours = [((ii / count * 360.0), 0.8, 0.8)
                           for ii in range(count)]
            colours = [colorsys.hsv_to_rgb(*cc) for cc in hsv_colours]

        edge_route_counts = defaultdict(int)
        edge_offsets = {}

        map_size = (np_city_pos.max(axis=0)[0] - np_city_pos.min(axis=0)[0])
        offset_step_size = map_size.mean() * args.os

        if data.fixed_routes is not None:
            fixed_routes = data.fixed_routes.squeeze(0)
            n_fixed_routes = fixed_routes.size(0)
            routes = list(fixed_routes) + list(routes)
            colours = (['orange'] * n_fixed_routes) + colours
        else:
            n_fixed_routes = 0

        for ri, (route, colour) in enumerate(zip(routes, colours)):
            if ri < n_fixed_routes:
                # draw the fixed route nodes as circles
                poss = np_city_pos[route]
                markersize = 200
                plt.scatter(poss[:, 0], poss[:, 1], color=colour, s=markersize)

            route_len = (route > -1).sum()
            route = route[:route_len]

            # update the counts of routes on each edge, and calculate the offset
            for si in range(len(route) - 1):
                edge = (route[si].item(), route[si + 1].item())
                # ensure the edge is always in the same order
                edge = (min(edge), max(edge))
                offset_count = edge_route_counts[edge]

                if edge not in edge_offsets:
                    # compute the edge's offset
                    ii = min(edge)
                    jj = max(edge)
                    edge_dir = (np_city_pos[ii] - np_city_pos[jj])
                    # perpendicular to the edge
                    base_offset = np.array([edge_dir[1], -edge_dir[0]])
                    base_offset = standardize_dir(base_offset)
                    edge_offsets[edge] = base_offset * offset_step_size
                else:
                    base_offset = edge_offsets[edge]
                edge_route_counts[edge] += 1

                # draw this edge of the route
                offset = base_offset * offset_count
                start_pos = np_city_pos[edge[0]]
                end_pos = np_city_pos[edge[1]]
                if si > 0:
                    prev_pos = np_city_pos[route[si - 1]]
                else:
                    prev_pos = None
                start = get_offset_corner(start_pos, end_pos, 
                                          prev_pos, offset, True)
                if si < len(route) - 2:
                    next_pos = np_city_pos[route[si + 2]]
                else:
                    next_pos = None
                end = get_offset_corner(start_pos, end_pos, next_pos, offset,
                                        False)
                if ri < n_fixed_routes:
                    width = args.es * 8
                else:
                    width = args.es
                plt.plot((start[0], end[0]), (start[1], end[1]), color=colour, 
                         linewidth=width)
        
        csv_str += "," + str(len(edge_route_counts))
        print(csv_str)

        for edge, count in edge_route_counts.items():
            if count > 1:
                print(f"edge {edge} has {count} routes")
            
    x_min, y_min = np_city_pos.min(axis=0)
    x_max, y_max = np_city_pos.max(axis=0)
    pu.set_tufte_spines(ax, x_min, x_max, y_min, y_max)
    plt.axis("on")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.set_aspect('equal', adjustable='box')
    plt.xlabel("Meters")
    # plt.tight_layout()
    if args.o:
        plt.savefig(args.o)
        plt.clf()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="dataset config path")
    parser.add_argument("--routes", help=".pkl file with generated routes")
    parser.add_argument("--demand", action="store_true", 
                        help="show the demand on the graph")
    parser.add_argument("--ns", "--node_size", type=float, default=10,
                        help="size of displayed nodes")
    parser.add_argument("--es", "--edge_size", type=float, default=10,
                        help="thickness of displayed edges")
    parser.add_argument("--os", type=float,
                        default=0.0015, help="scale factor for edge offsets")
    parser.add_argument("--height", type=float, help="figure height")
    parser.add_argument("--width", type=float, help="figure width")
    parser.add_argument('--special_coords', '--sc', 
                        help="A yaml file with special coordinates to draw")
    parser.add_argument('-o', help='If provided, save to file')
    parser.add_argument('--label_special', '--ls', action='store_true',
                        help='Label the special nodes')
    args = parser.parse_args()

    if args.height or args.width:
        base_size = plt.rcParams['figure.figsize']
        if args.height and args.width:
            size = (args.width, args.height)
        elif args.width:
            size = (args.width, base_size[1])
        elif args.height:
            size = (base_size[0], args.height)
        plt.rcParams["figure.figsize"] = size

    ds_cfg = omegaconf.OmegaConf.load(args.dataset)
    if 'dataset' in ds_cfg:
        # the dataset is nested in the eval config, so pull it out
        ds_cfg = ds_cfg.dataset
    
    data = get_dataset_from_config(ds_cfg, center_nodes=False)[0]

    if args.routes is None:
        # just plot the street graph
        plot_routes(args, data)

    else:
        # load the routes
        routes = tu.load_routes_tensor(args.routes)
        with open(args.routes, "rb") as ff:
            unloaded = pickle.load(ff)

            for routes in unloaded:
                plot_routes(args, data, routes)


if __name__ == "__main__":
    main()