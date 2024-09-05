import networkx as nx
import pytest
from learning.utils import yen_k_shortest_paths

@pytest.fixture
def simple_graph():
    graph = nx.Graph()
    graph.add_weighted_edges_from([
        (0, 1, 1), (1, 2, 1.5), (0, 2, 2),
        (2, 3, 1), (1, 3, 2), (3, 4, 1),
        (1, 4, 5)
    ])
    return graph

def test_single_path(simple_graph):
    expected_path = [[0, 1, 3, 4]]
    result = yen_k_shortest_paths(simple_graph, 0, 4, 1)
    assert result == expected_path

def test_multiple_paths(simple_graph):
    expected_paths = [[0, 1, 3, 4], [0, 2, 3, 4]]
    result = yen_k_shortest_paths(simple_graph, 0, 4, 2)
    assert result == expected_paths

def test_no_path(simple_graph):
    # Remove destination to create a no path scenario
    simple_graph.remove_node(4)
    with pytest.raises(nx.NetworkXNoPath):
        yen_k_shortest_paths(simple_graph, 0, 4, 2)

def test_large_kk_value(simple_graph):
    # More paths requested than available
    result = yen_k_shortest_paths(simple_graph, 0, 4, 10)  
    # Check if it does not exceed the number of unique paths    
    assert len(result) <= 10  

def test_same_source_and_destination(simple_graph):
    expected_path = [[0]]
    result = yen_k_shortest_paths(simple_graph, 0, 0, 1)
    assert result == expected_path

    result = yen_k_shortest_paths(simple_graph, 0, 0, 3)
    assert result == expected_path
