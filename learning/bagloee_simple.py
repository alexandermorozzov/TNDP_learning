import logging as log

from tqdm import tqdm
import torch
from omegaconf import DictConfig
from torch_geometric.data import DataLoader
import hydra

from torch_utils import get_batch_tensor_from_routes
from learning import bagloee as bgl
from simulation.citygraph_dataset import get_dataset_from_config, STOP_KEY
import learning.utils as lrnu


class DummySim:
    def __init__(self, city_graph):
        self.data = city_graph

    def get_basin_walk_times_matrix(self, device=None):
        """This is just the identity matrix, since demand nodes aren't 
            separate from stop nodes."""
        if device is None:
            device = self.device
        basin_mat = torch.eye(self.data[STOP_KEY].num_nodes, device=device)
        basin_mat[basin_mat == 0] = float('inf')
        basin_mat.fill_diagonal_(0)
        return basin_mat
    
    def get_street_edge_matrix(self, device=None):
        if device is None:
            device = self.device
        # remove the batch dimension
        return self.data.street_adj[0].to(device)
    
    def get_demand_matrix(self, device=None):
        if device is None:
            device = self.device
        # remove the batch dimension
        return self.data.demand[0].to(device)
    
    @property
    def drive_times(self):
        return self.data.drive_times[0]
    
    @property
    def drive_dists(self):
        return self.data.drive_dists[0]
        
    @property
    def device(self):
        return self.data[STOP_KEY].x.device
    
    @property
    def n_street_nodes(self):
        return self.data[STOP_KEY].num_nodes
    
    @property
    def cfg(self):
        return {
            'mean_stop_time_s': 0,
        }


def bagloee_simple(graph_data, n_routes, min_route_len, max_route_len, 
                   cost_obj, sum_writer=None, silent=False, 
                   symmetric_routes=True, device=None, **bagloee_kwargs):
    sim = DummySim(graph_data)

    # generate candidates
    candidate_routes = bgl.get_newton_routes(
        sim, 
        bagloee_kwargs['alpha'], 
        bagloee_kwargs['dsp_step'],
        bagloee_kwargs['min_access_egress'],
        bagloee_kwargs['max_dsp_threshold'],
        bagloee_kwargs['dsp_feeder_threshold'],
        bagloee_kwargs['dsp_local_threshold'],
        bagloee_kwargs['min_route_length_m'],
        all_routes_are_loops=symmetric_routes,
        device=device)
    
    # filter candidates that aren't within the length limits
    candidate_routes = [route for route in candidate_routes
                        if min_route_len <= len(route) <= max_route_len]
    
    # run the main algorithm GA-ACO algorithm
    routes, cost_history = bagloee_route_selection(
        graph_data, cost_obj, candidate_routes, n_routes, 
        bagloee_kwargs['termination_limit'], sum_writer, device)
    return get_batch_tensor_from_routes(routes, device), cost_history


def bagloee_route_selection(
        graph_data, cost_obj, candidate_routes, n_routes,
        termination_limit, sum_writer=None, device=None):
    """
    Route selection and frequency setting as in Bagloee & Ceder 2011.

    arguments
    candidate_routes -- a list of candidate bus routes
    n_routes -- the number of routes to include
    termination_limit -- the number of phase 1 iterations with no improvement 
        after which to terminate.
    sum_writer -- a tensorboard summary writer.  If provided, quality will be
        logged as the algorithm progresses.
    """
    # set up variables for the loop

    # compute duration of period being analyzed, in seconds
    points = []
    n_cdts = len(candidate_routes)
    use_counts = torch.zeros(n_cdts, dtype=int)
    past_scenarios = []
    top_qual_and_idxs = []
    cost_history = []

    scen_tracker = bgl.TotalInfoTracker(len(candidate_routes), device)

    def get_weights(scenario):
        # the paper doesn't describe this weighting by selection count, but
         # it makes the algorithm run about as fast as they report, and doesn't
         # hurt performance.
        weights = 1 / (use_counts + 1)
        weights[scenario] = 0
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights /= weights.sum()
        return weights

    phase = 0
    log.info(f"termination limit is {termination_limit}")
    ii = 1
    pbar = tqdm()
    while True:
        scenario = []
        if ii <= 6 or not (use_counts > 0).all():
            # we're in phase 0
            # Build a random scenario, upweighting unused routes
            weights = get_weights(scenario)
            while len(scenario) < n_routes and weights.sum() > 0:
                # randomly pick a route from the candidates
                route_idx = weights.multinomial(1).item()
                # route_idx = .random.choice(n_cdts, p=weights)
                # add the route to the scenario
                scenario.append(route_idx)
                # set weight to zero to avoid double-picking
                weights = get_weights(scenario)
                if weights.sum() == 0 and set(scenario) in past_scenarios and \
                     torch.rand(1) < 0.1:
                    # this scenario was already used.  Mutate it!
                    bgl.pop_random(scenario)
                    weights = get_weights(scenario)

            use_counts[scenario] += 1
            n_unused = n_cdts - (use_counts > 0).sum()
            log.debug(f"{n_unused} candidate routes have not been used")
                
        else:
            # we're in phase 1
            if phase == 0:
                log.info(f"phase 0 lasted {ii - 1} iterations")
                steps_since_improvement = 0
                phase = 1

            # compute the collective points (equation 34)
            top_6_quals, top_6_idxs = zip(*top_6_qual_and_idxs)
            top_6_points = torch.stack([points[ii] for ii in top_6_idxs])
            proportions_of_best = (torch.cat(top_6_quals) / 
                                   top_6_quals[5])**2
            weights = torch.rand(6, device=device) * proportions_of_best
            clctv_pts = (weights[:, None] * top_6_points).sum(axis=0)

            # for route in sorted routes:
            sorted_route_idxs = sorted(range(n_cdts), 
                                       key=lambda ri: clctv_pts[ri], 
                                       reverse=True)
            scenario = sorted_route_idxs[:n_routes]

            mut_count = 0
            while set(scenario) in past_scenarios and mut_count < 5 * n_cdts:
                # log.info("mutating scenario...")
                # mutate scenario as above
                if len(scenario) > 0:
                    _, _ = bgl.pop_random(scenario)
                    
                is_valid = torch.ones(n_cdts, dtype=bool, device=device)
                is_valid[scenario] = False
                if not is_valid.any():
                    log.warning("phase 1 generating empty scenarios repeatedly!")
                    break
                route_dstrb = is_valid / is_valid.sum()
                chosen_idx = route_dstrb.multinomial(1).item()
                scenario.append(chosen_idx)
                mut_count += 1
            if len(scenario) > 0:
                use_counts[scenario] += 1

        scen_tracker.add_scenario(scenario)

        # simulate the proposed scenario
        scen_routes = [candidate_routes[ri] for ri in scenario]
        # get the objective function value
        cost, per_route_riders = \
            cost_obj(scen_routes, graph_data, return_per_route_riders=True)
        quality = -cost
        cost_history.append(cost)
            
        if sum_writer:
            sum_writer.add_scalar('n scenarios seen', 
                                  scen_tracker.n_scenarios_so_far, ii-1)
            sum_writer.add_scalar('total exploration', 
                                  scen_tracker.total_info(), ii-1)
            sum_writer.add_scalar('mean cost', cost, ii-1)

        # update the top six qualities
        if len(top_qual_and_idxs) < 6:
            # we have less than six, so add each one
            top_qual_and_idxs.append((quality, ii-1))
            if len(top_qual_and_idxs) == 6:
                # we've reached six, so sort them
                top_6_qual_and_idxs = sorted(top_qual_and_idxs)
        elif quality > top_6_qual_and_idxs[0][0]:
            # this scenario should be in the top 6, so remove old 6th-best
            top_6_qual_and_idxs.pop(0)
            # determine this scenario's placement in top 6
            rank_in_top6 = 0
            for t6q, _ in top_6_qual_and_idxs:
                if quality > t6q:
                    rank_in_top6 += 1
                else:
                    break
            top_6_qual_and_idxs.insert(rank_in_top6, (quality, ii-1))

        if phase == 1:
            if quality < top_6_qual_and_idxs[5][0]:
                # we haven't improved...
                steps_since_improvement += 1
                log.debug(f"steps since improvement: {steps_since_improvement}")
            else:
                steps_since_improvement = 0
                log.info(f"best scenario updated with quality {quality}")
                if sum_writer:
                    # log new best
                    sum_writer.add_scalar('best quality', quality, ii-1)

        # update points (equation 28)
        n_riders = per_route_riders.sum()
        if len(points) == 0:
            points_i = torch.zeros(n_cdts, device=device)
        else:
            points_i = points[-1].clone()
        # make denominator at least 1 to avoid division by 0
        points_i[scenario] = quality * per_route_riders / max(n_riders, 1)
        assert not torch.isnan(points_i).any()
        points.append(points_i)

        # save this iteration's scenario, even if it's a duplicate
        set_scenario = set(scenario)
        past_scenarios.append(set_scenario)
        
        if phase == 1 and steps_since_improvement == termination_limit:
            # no improvement for a while, so terminate.
            break

        ii += 1
        pbar.update(1)

    pbar.close()
    log.info(f"{ii} iterations were run")

    # return the best scenario routes and frequencies
    _, best_idx = top_6_qual_and_idxs[-1]
    best_scenario = list(past_scenarios[best_idx])
    routes = [candidate_routes[ri] for ri in best_scenario]
    
    return routes, torch.stack(cost_history)


@hydra.main(version_base=None, config_path="../cfg", 
            config_name="bagloee_mumford")
def main(cfg: DictConfig):
    global DEVICE

    DEVICE, run_name, sum_writer, cost_fn, _ = \
        lrnu.process_standard_experiment_cfg(cfg, 'bagloee_')

    # read in the dataset
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    draw = cfg.eval.get('draw', False)

    all_route_sets = []

    for n_routes in cfg.eval.n_routes:
        routes = lrnu.test_method(
            bagloee_simple, test_dl, n_routes, cfg.eval.min_route_len, 
            cfg.eval.max_route_len, cost_fn, sum_writer=sum_writer, 
            silent=False, symmetric_routes=cfg.experiment.symmetric_routes,
            draw=draw, device=DEVICE, return_routes=True, **cfg.alg_kwargs)[-1]
        if type(routes) is not torch.Tensor:
            routes = get_batch_tensor_from_routes(routes)
        all_route_sets.append(routes.cpu())
    # save the final routes that were produced
    lrnu.dump_routes(run_name, cfg.eval.n_routes, all_route_sets)


if __name__ == '__main__':
    main()