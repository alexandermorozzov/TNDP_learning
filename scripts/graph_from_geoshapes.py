import argparse
from pathlib import Path
import logging as log
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
import pyproj
import matplotlib.pyplot as plt
from shapely.geometry import Point

from simulation.citygraph_dataset import CityGraphData


def collect_demand_at_polygons(demands_df, gdf, polygon_frame):
    # build transformer from latlon to polygon frame
    tf = pyproj.Transformer.from_crs("EPSG:4326", polygon_frame)
    n_polys = len(gdf)
    demand_matrix = np.zeros((n_polys, n_polys))
    # correlate demands with census tracts by their CTIDs
    for _, row in demands_df.iterrows():
        origin = (row['t_orilat'], row['t_orilon'])
        destination = (row['t_deslat'], row['t_deslon'])
        demand = row['t_expf']
        tfed_orig = tf.transform(*origin)
        contains_orig = gdf.geometry.contains(Point(*tfed_orig))
        origin_is_valid = contains_orig.any()
        tfed_dest = tf.transform(*destination)
        contains_dest = gdf.geometry.contains(Point(*tfed_dest))
        dest_is_valid = contains_dest.any()
        if origin_is_valid and dest_is_valid:
            orig_poly_idx = contains_orig.argmax()
            dest_poly_idx = contains_dest.argmax()
            if orig_poly_idx != dest_poly_idx:
                demand_matrix[orig_poly_idx, dest_poly_idx] += demand

    if not np.isclose(demand_matrix, demand_matrix.T).all():
        # force demand matrix to be symmetric
        log.warning("Demand matrix is not symmetric")
        demand_matrix = np.maximum(demand_matrix, demand_matrix.T)

    return demand_matrix


def build_graph(census_tract_path, streets_path, demand_csv_path, 
                city_name, census_frame='EPSG:3347', 
                streets_frame='EPSG:32188'):
    # build a graph from the census tracts that border each other
    gdf = gpd.read_file(census_tract_path)
    zone_graph = nx.DiGraph()
    for tract_idx in gdf.index:
        zone_graph.add_node(tract_idx)
    for index, ct in gdf.iterrows():
        # stop this from adding new nodes
        neighbours = ~gdf.geometry.disjoint(ct.geometry)
        for joint_index, is_neighbour in neighbours.items():
            if is_neighbour:
                zone_graph.add_edge(index, joint_index)

    # load demand
    log.info('process demand')
    demands_df = pd.read_csv(demand_csv_path, 
                             converters={'t_orict': str, 't_desct': str})
    demand_matrix = collect_demand_at_polygons(demands_df, gdf, census_frame)

    # determine driving times between census tracts
    log.info('load the network graph')
    nx_road_graph = nx.read_shp(str(streets_path))
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

    # add edge drive times to all edges as "weight"
    for src, dst, data in nx_road_graph.edges(data=True):
        speed_kph = data['VitesseMAX']
        length_m = data['Shape_Leng']
        time_s = length_m / (speed_kph / 3.6)
        nx_road_graph[src][dst]['weight'] = time_s

    # compute all inter-node drivetimes
    inter_node_drivetimes = defaultdict(dict)
    print('compute all inter-node drivetimes')
    for src in tqdm(nx_road_graph.nodes()):
        drivetimes, _ = nx.multi_source_dijkstra(nx_road_graph, {src})
        for dst, drivetime in drivetimes.items():
            inter_node_drivetimes[src][dst] = drivetime

    # assign each node to a census tract
    print('assign nodes to census tracts')
    tract_nodes = defaultdict(list)
    tf = pyproj.Transformer.from_crs(streets_frame, census_frame)
    # find all drive distances between street nodes

    for point in nx_road_graph.nodes():
        # apply the appropriate transformation to the point
        street_node_loc = Point(*tf.transform(*point))
        tract_idx = gdf.geometry.contains(street_node_loc).argmax()
        # append the un-transformed points, because they are the node ids
         # in the graph
        tract_nodes[tract_idx].append(point)

    # for each pair of tracts:
    tract_pair_dt_lists = defaultdict(list)
    print('compute all inter-tract drivetimes')
    for tract1, tract2 in tqdm(zone_graph.edges()):
        if tract1 == tract2:
            tract_pair_dt_lists[(tract1, tract2)].append(0)
            continue
        for node1 in tract_nodes[tract1]:
            for node2 in inter_node_drivetimes[node1]:
            # for node2 in tract_nodes[tract2]:
                # add the drive time between nodes to the tract-pair's collection
                drive_time = inter_node_drivetimes[node1][node2]
                tract_pair_dt_lists[(tract1, tract2)].append(drive_time)
                
    # compute the average drive time between each tract pair
    for (tract1, tract2), drive_times in tract_pair_dt_lists.items():
        median_drive_time_m = np.median(drive_times) / 60
        zone_graph[tract1][tract2]['weight'] = median_drive_time_m

    # convert this into a weighted edge matrix and write to a file
    edge_time_mat = nx.to_numpy_array(zone_graph, weight='weight', 
                                      nonedge=np.inf)
    # force the drive-time matrix to be symmetric, as below
    assert not ((edge_time_mat == np.inf) ^ (edge_time_mat.T == np.inf)).any()
    edge_time_mat = np.maximum(edge_time_mat, edge_time_mat.T)

    components = nx.strongly_connected_components(zone_graph)
    components = list(components)
    if len(components) > 1:
        log.warning(f"found {len(components)} connected components")
        largest_component = list(max(components, key=len))
        edge_time_mat = \
            edge_time_mat[largest_component][:, largest_component]
        demand_matrix = demand_matrix[largest_component][:, largest_component]
        zone_graph = zone_graph.subgraph(largest_component)

    n_nodes = len(zone_graph.nodes())
    log.info(f"number of nodes: {n_nodes}")

    # get centroids as node locations, and write to a file
    with open(city_name + 'Coords.txt', 'w') as ff:
        ff.write(f"{n_nodes}\n")
        for node in zone_graph.nodes():
            row = gdf.iloc[node]
            zone_centroid = row.geometry.centroid
            ff.write(f"{zone_centroid.x} {zone_centroid.y}\n")

    # write the demand matrix to a file
    np.savetxt(city_name + 'Demand.txt', demand_matrix, fmt='%.1f')

    # write travel times
    np.savetxt(city_name + 'TravelTimes.txt', edge_time_mat, fmt='%.1f')
    

def point_list_to_nparray(point_list):
    return np.array([[point.x, point.y] for point in point_list])


def main():
    # get the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument('poly_path', type=str,
                        default='../laval/dissemination_areas/laval_dissemination_areas.shp',
                        help='path to the shapefile with region polygons')
    parser.add_argument('streets_path', type=str,
                        help='path to the streets shapefile')
    parser.add_argument('dmd_path', type=str, default='../laval/od_2013.csv',
                        help='path to the demand matrix')
    parser.add_argument('-n', default='Laval', 
                        help='name of city for output files.')
    args = parser.parse_args()

    build_graph(args.poly_path, args.streets_path, args.dmd_path, args.n)

    cgd = CityGraphData.from_mumford_data('.', args.n, 
                                          scale_dynamically=False)
    cgd.draw()
    plt.show()

    
if __name__ == '__main__':
    main()