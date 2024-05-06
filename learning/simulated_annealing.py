import math
import logging as log

from tqdm import tqdm
import torch
from torch_geometric.loader import DataLoader
import hydra
from omegaconf import DictConfig

from learning.initialization import get_direct_sat_dmd
import learning.utils as lrnu
from torch_utils import reconstruct_all_paths, get_unselected_routes, \
    dump_routes
from simulation.citygraph_dataset import get_dataset_from_config
from bee_colony import get_bee_1_variants, get_bee_2_variants



def simulated_annealing_with_reheating(state, cost_obj, init_network, 
                                       initial_temp, final_temp, n_iterations, 
                                       schedule, cooling_rate, shorten_prob,
                                       reheating_threshold=None, 
                                       reheating_factor=None, sum_writer=None, 
                                       silent=False):
    """Performs the simulated annealing algorithm with reheating.

    Args:
        state: The initial state of the system, not including the routes.
        cost_obj: The cost-function object.
        initial_temp: The initial temperature.
        final_temp: The final temperature.
        n_iterations: The number of iterations to run the algorithm for.
        schedule: The temperature schedule to use.
        cooling_rate: The cooling rate to use.
        reheating_threshold: The number of iterations without improvement 
            before reheating.
        reheating_factor: The factor by which to increase the temperature when
            reheating.
        shorten_prob: The probability of shortening a route in the type-2
            mutator.
        sum_writer: The summary writer to use for logging.
        silent: Whether to suppress output.
        init_network: The initial network to start from.

    Returns:
        The best solution found.
    """
    dev = state.device
    # get all shortest paths
    shortest_paths, _ = reconstruct_all_paths(state.nexts)
    demand = torch.nn.functional.pad(state.demand, (0, 1, 0, 1))

    current_network = init_network

    def get_cost(network):
        state.replace_routes(network)
        return cost_obj(state).cost

    best_network = current_network 
    best_cost = get_cost(best_network)
    cost_norm = best_cost
    current_cost = best_cost.clone()
    current_temp = initial_temp
    no_improvement_iterations = 0
    if reheating_threshold is None:
        # we never reheat
        reheating_threshold = n_iterations

    cost_history = torch.zeros((state.batch_size, n_iterations+1), device=dev)
    cost_history[0] = current_cost
    if sum_writer is not None:
        sum_writer.add_scalar('cost', current_cost, 0)
        sum_writer.add_scalar('best cost', best_cost, 0)
        sum_writer.add_scalar('temperature', current_temp, 0)

    # set up some matrices that will be used to choose modifications
    direct_sat_dmd = get_direct_sat_dmd(demand, shortest_paths,
                                        cost_obj.symmetric_routes)
    street_node_neighbours = (state.street_adj.isfinite() &
                              (state.street_adj > 0))
    
    for ii in tqdm(range(n_iterations), disable=silent):
        new_network = modify(state, current_network, direct_sat_dmd,
                             shorten_prob, street_node_neighbours, 
                             shortest_paths, dev)
        new_cost = get_cost(new_network)
        cost_diff = new_cost - current_cost
        
        accept_worse_prob = math.exp(-cost_diff / (cost_norm * current_temp))
        if cost_diff < 0 or torch.rand(1) < accept_worse_prob:
            current_network = new_network
            current_cost = new_cost
            if current_cost < best_cost:
                best_network = current_network
                best_cost = current_cost
                no_improvement_iterations = 0
            else:
                no_improvement_iterations += 1
        else:
            no_improvement_iterations += 1
        
        current_temp = max(schedule(current_temp, cooling_rate), 
                           final_temp)

        # Reheating condition
        if no_improvement_iterations > reheating_threshold:
            current_temp = min(reheating_factor * current_temp, 
                               initial_temp)
            no_improvement_iterations = 0  # Reset counter after reheating

        cost_history[:, ii + 1] = current_cost
        if sum_writer is not None:
            sum_writer.add_scalar('cost', current_cost, ii+1)
            sum_writer.add_scalar('best cost', best_cost, ii+1)
            sum_writer.add_scalar('temperature', current_temp, ii+1)
            sum_writer.add_scalar('worsen prob', min(accept_worse_prob, 1), ii)

    state.replace_routes(best_network)
    return state, cost_history


# Temperature Schedules
def exponential_cooling(current_temp, rate):
    return current_temp * rate

def linear_cooling(current_temp, rate):
    return current_temp - rate


def modify(state, network, direct_sat_dmd, shorten_prob, 
           street_node_neighbours, shortest_paths, device):
    """Modify the current solution by adding or removing a random connection."""
    # First pass: no hyperheuristic, just choose one at random
    # TODO try a hyperheuristic?
    new_network = network.clone()
    # choose a random route to modify
    route_idx = torch.randint(network.shape[-2], (1,), device=device)
    chosen_route = network[..., route_idx, :]
    
    if torch.rand(1) < 0.5:
        # apply type-1 mutator
        # add a "bee" dimension to route_idx, as get_unselected_routes needs it
        unsel_routes = get_unselected_routes(network, route_idx)
        remaining_state = state.clone()
        remaining_state.replace_routes(unsel_routes)
        new_route = get_bee_1_variants(remaining_state, chosen_route,
                                       direct_sat_dmd, shortest_paths)
    else:
        # apply type-2 mutator
        new_route = get_bee_2_variants(chosen_route, shorten_prob, 
                                       street_node_neighbours)
        
    new_network[..., route_idx, :] = new_route    
    return new_network


@hydra.main(version_base=None, config_path="../cfg", config_name="sa_linear")
def main(cfg: DictConfig):
    if cfg['alg_args']['schedule'] == 'exponential':
        schedule = exponential_cooling
    elif cfg['alg_args']['schedule'] == 'linear':
        schedule = linear_cooling
    else:
        raise ValueError(f"Invalid temperature schedule {cfg['schedule']}")

    global DEVICE
    DEVICE, run_name, sum_writer, cost_fn, _ = \
        lrnu.process_standard_experiment_cfg(cfg, 'sa_', weights_required=False)

    # read in the dataset
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=1)
        
    sa_kwargs = dict(cfg['alg_args'])
    sa_kwargs['schedule'] = schedule

    routes = lrnu.test_method(simulated_annealing_with_reheating, test_dl, 
                              cfg.eval, cfg.init, cost_fn, 
                              sum_writer=sum_writer, device=DEVICE, 
                              return_routes=True, silent=False, **sa_kwargs)[-1]

    # save the final routes that were produced
    dump_routes(run_name, routes)


# Example usage
if __name__ == "__main__":
    main()