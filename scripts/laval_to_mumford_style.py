import argparse
import logging as log
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import networkx as nx
import pandas as pd
import pyproj
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph, RemoveIsolatedNodes
import torch_geometric.utils as pygu

from simulation.citygraph_dataset import CityGraphData


def from_census_tracts(demand_csv_path, mapframe="EPSG:32188"):
    demands_df = pd.read_csv(demand_csv_path)
    locs_in_census_tract = defaultdict(list)
    tf = pyproj.Transformer.from_crs("EPSG:3348", mapframe)
    for _, row in demands_df.iterrows():
        orig = tf.transform(row['t_orix'], row['t_oriy'])
        locs_in_census_tract[row['t_orict']].append((orig))
        dest = tf.transform(row['t_desx'], row['t_desy'])
        locs_in_census_tract[row['t_desct']].append((dest))

    centroids = {}
    for ct, poss in locs_in_census_tract.items():
        centroids[ct] = np.mean(poss, axis=0)

    # TODO resume here.
    

def from_xml_and_csv(shapefile_path, gtfs_path, demand_csv_path, 
                     basin_radius_m=500, beeline_dist_factor=1.3, 
                     edge_threshold_minutes=5, mapframe="EPSG:32188"):
    # load stops
    stops_path = Path(gtfs_path) / "stops.txt"
    stops_df = pd.read_csv(stops_path)
    log.info(f"# original stops: {len(stops_df)}")
    stop_locs = []

    # collect the stops from the gtfs file
    pure_names_to_locs = defaultdict(list)
    for _, row in stops_df.iterrows():
        name_wo_code = row["stop_name"].partition("[")[0]
        pure_name = name_wo_code.partition(" Quai")[0]
        loc = (row["stop_lat"], row["stop_lon"])
        pure_names_to_locs[pure_name].append(loc)
    # transform the stop locs from lat,lon to the map frame
    transformer = pyproj.Transformer.from_crs("EPSG:4326", mapframe)
    pure_names_to_locs = {nn: transformer.transform(*np.mean(ll, axis=0))
                          for nn, ll in pure_names_to_locs.items()}
    _, stop_locs = zip(*pure_names_to_locs.items())

    log.info('load the network graph')
    nx_road_graph = nx.read_shp(shapefile_path)
    # filter out 0 speeds
    # make ONEWAY_BUS=None edges bidirectional
    to_remove = []
    to_add = []
    for src, dst, data in list(nx_road_graph.edges(data=True)):
        if data['VitesseMAX'] == 0 or data['ONEWAY_BUS'] == "N":
            to_remove.append((src, dst))
        elif data['ONEWAY_BUS'] == "TF":
            # flip the edge so it points in the driving direction
            to_remove.append((src, dst))
            to_add.append((dst, src, data))
        elif data['ONEWAY_BUS'] is None:
            # add edge in the other direction, since it's bidirectional
            to_add.append((dst, src, data))
    nx_road_graph.remove_edges_from(to_remove)
    nx_road_graph.add_edges_from(to_add)

    # add drive times to all edges
    for src, dst, data in nx_road_graph.edges(data=True):
        speed_kph = data['VitesseMAX']
        length_m = data['Shape_Leng']
        time_s = length_m / (speed_kph / 3.6)
        nx_road_graph[src][dst]['drivetime'] = time_s

    node_locs = np.array(nx_road_graph.nodes())
    
    # map the stop locations onto network node locations
    stop_to_node_dists = cdist(stop_locs, node_locs)
    # remove duplicate node-stops
    closest_nodes = np.unique(stop_to_node_dists.argmin(axis=1))
    stop_locs = node_locs[closest_nodes]
    log.info(f"# real stop locations: {len(closest_nodes)}")

    # TODO filter the stops a la Bagloee?

    # We assume asymmetry here
    # filter out nodes that don't correspond to a stop, and bridge their edges
    log.info('build the travel time matrix')
    # essentially, relabel nodes with index numbers
    edge_drive_times = nx.to_numpy_array(nx_road_graph, weight="drivetime",
                                         nonedge=0)
    edt_graph = nx.from_numpy_array(edge_drive_times, create_using=nx.DiGraph)
    # TODO the networkx graph produced is fully-connected.  Fix that.
    stop_edge_times = np.full_like(edge_drive_times, float('inf'))
    np.fill_diagonal(stop_edge_times, 0)
    for src in tqdm(closest_nodes):
        lengths, src_paths = nx.multi_source_dijkstra(edt_graph, {src})

        for dst, path in src_paths.items():
            if dst not in closest_nodes:
                continue

            # if the path does not go through another stop, add an edge
            if not any([node_idx in closest_nodes for node_idx in path[1:-1]]):
                stop_edge_times[src, dst] = lengths[dst]

    # cut the nodes that got bridged
    stop_edge_times = stop_edge_times[closest_nodes][:, closest_nodes]
    # expected time units for this file are in minutes
    stop_edge_times /= 60
    stop_edge_times[stop_edge_times > edge_threshold_minutes] = float('inf')
    # # instead of thresholding, keep only the closest 6 edges
    # # this kinda works, but breaks some parts of the graph.
    # pyg_graph = Data(pos=torch.tensor(stop_locs))
    # knn_graph = KNNGraph(k=6, flow='source_to_target')
    # rmv_isolated_nodes = RemoveIsolatedNodes()
    # pyg_graph = rmv_isolated_nodes(knn_graph(pyg_graph))
    # nx_graph = pygu.to_networkx(pyg_graph, to_undirected=False)
    # import pdb; pdb.set_trace()
    # stop_edge_times = nx.to_numpy_array(nx_graph, weight="edge_attr")

    # remove any nodes that are disconnected
    nx_stop_edge_times = stop_edge_times.copy()
    nx_stop_edge_times[nx_stop_edge_times == float('inf')] = 0
    stop_graph = nx.from_numpy_array(nx_stop_edge_times, 
                                     create_using=nx.DiGraph)
    components = nx.strongly_connected_components(stop_graph)
    components = list(components)
    if len(components) > 1:
        log.warning(f"found {len(components)} connected components")
        largest_component = list(max(components, key=len))
        stop_edge_times = \
            stop_edge_times[largest_component][:, largest_component]
        stop_locs = stop_locs[largest_component]

    # get the demands between demand locations
    log.info('build the demand matrix')
    demands_df = pd.read_csv(demand_csv_path)
    demand_graph = nx.DiGraph()
    demand_poss = set()
    tf = pyproj.Transformer.from_crs("EPSG:3348", mapframe)
    for _, row in demands_df.iterrows():
        orig = tf.transform(row['t_orix'], row['t_oriy'])
        demand_poss.add(orig)
        dest = tf.transform(row['t_desx'], row['t_desy'])
        demand_poss.add(dest)
        if demand_graph.has_edge(orig, dest):
            demand_graph[orig][dest]['demand'] += row['t_expf']
        else:
            demand_graph.add_edge(orig, dest, demand=row['t_expf'])

    raw_demand = nx.to_numpy_array(demand_graph, weight="demand")
    # make demands symmetrical by taking the max in either direction
    raw_demand = np.maximum(raw_demand, raw_demand.T)
    np.fill_diagonal(raw_demand, 0)

    # allocate demand to stop nodes by some strategy
    demand_locs = np.array(demand_graph.nodes())
    basin_crowflies_dists = cdist(stop_locs, demand_locs)
    basin_dists = basin_crowflies_dists * beeline_dist_factor
    closest_stops = basin_dists.argmin(axis=0)
    min_dists = basin_dists.min(axis=0)
    no_basin_stop = min_dists > basin_radius_m
    n_uncovered = no_basin_stop.sum()
    if n_uncovered > 0:
        log.warning(f"{n_uncovered} of {len(raw_demand)} demand locations "\
                   "are not covered by a stop")
    stop_demands = np.zeros((len(stop_locs), len(stop_locs)))
    for row_idx, raw_dmd_row in tqdm(enumerate(raw_demand), 
                                     total=len(raw_demand)):
        from_stop_dist = basin_dists[closest_stops[row_idx], row_idx]
        if from_stop_dist > basin_radius_m:
            # no stops within basin of origin node, so skip this 
            continue

        for col_idx, raw_dmd_elem in enumerate(raw_dmd_row):
            to_stop_dist = basin_dists[closest_stops[col_idx], col_idx]
            if to_stop_dist > basin_radius_m:
                # no stops within basin of destination node, so skip this node
                continue

            src_stop_idx = closest_stops[row_idx]
            dst_stop_idx = closest_stops[col_idx]
            stop_demands[src_stop_idx, dst_stop_idx] += raw_dmd_elem

    # convert to the data format we need
    log.info('write text files')
    # first, write the node coordinates
    with open('LavalCoords.txt', 'w') as ff:
        ff.write(f"{len(stop_locs)}\n")
        for stop_loc in stop_locs:
            ff.write(f"{stop_loc[0]} {stop_loc[1]}\n")

    np.savetxt('LavalTravelTimes.txt', stop_edge_times, fmt='%.1f')

    np.savetxt('LavalDemand.txt', stop_demands, fmt='%d')


def main():
    parser = argparse.ArgumentParser()
    # map_xml_path="/usr/local/data/ahollid/laval/network.xml",
    # gtfs_path="/usr/local/data/ahollid/laval/gtfs_nov2013",
    # demand_csv_path="/usr/local/data/ahollid/laval/od_2013.csv"
    parser.add_argument('shapefile_path', type=str)
    parser.add_argument('gtfs_path', type=str)
    parser.add_argument('demand_csv_path', type=str)
    parser.add_argument('--br', '--basin-radius', type=float, default=500,
                        help='basin radius in meters')
    parser.add_argument('--bdf', '--beeline-dist-factor', type=float, 
                        default=1.3)
    parser.add_argument('--et', '--edge-threshold', type=float, default=5,
                        help='edge threshold in minutes')
    args = parser.parse_args()

    log.basicConfig(level=log.INFO)
    from_xml_and_csv(args.shapefile_path,
                     args.gtfs_path,
                     args.demand_csv_path,
                     basin_radius_m=args.br,
                     beeline_dist_factor=args.bdf,
                     edge_threshold_minutes=args.et,
                     )

    cgd = CityGraphData.from_mumford_data('.', 'Laval', 
                                          scale_dynamically=False)
    cgd.draw()
    plt.show()


if __name__ == "__main__":
    main()