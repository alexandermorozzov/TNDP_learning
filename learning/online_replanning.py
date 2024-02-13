import time
import math
import logging as log

from tqdm import tqdm
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Batch

import learning.utils as lrnu
from simulation.citygraph_dataset import get_dataset_from_config
from learning.bee_colony import bee_colony
from learning.eval_route_generator import sample_from_model


class Planner:
    def __init__(self, cfg, cost_obj):
        self.cfg = cfg
        self.n_routes = cfg.eval.n_routes
        self.min_route_len = cfg.eval.min_route_len
        self.max_route_len = cfg.eval.max_route_len
        self.cost_obj = cost_obj
        self.allowed_time = cfg.iter_time_limit_s

    def get_update_size(self):
        # return the "size" of the update, which differs in meaning depending
         # on the type of planner.
        raise NotImplementedError()

    def set_update_size(self):
        # set the "size" of the update, which differs in meaning depending
         # on the type of planner.
        raise NotImplementedError()

    @property
    def cost_weights(self):
        return self.cost_obj.get_weights()

    def update_plan(self, plan, city_graph):
        start_time = time.perf_counter()
        updated_plan = self._update_plan_inner(plan, city_graph)
        duration = time.perf_counter() - start_time
        if duration > self.allowed_time:
            log.warning(f"Planning took {duration} seconds, which is longer "\
                        f"than the allowed time of {self.allowed_time} "\
                        f"seconds!")
        return updated_plan, duration
    
    def _update_plan_inner(self, plan, city_graph):
        # construction planner just generates a new route
        # BCO planner runs some BCO iterations
        # NBCO does the same...but they should be limited by time
        raise NotImplementedError()
    

class NeuralConstructionPlanner(Planner):
    def __init__(self, model, cfg, cost_obj):
        super().__init__(cfg, cost_obj)
        self.n_samples = cfg.online_planner.neural.n_samples
        self.model = model

    def get_update_size(self):
        return self.n_samples
    
    def set_update_size(self, n_samples):
        self.n_samples = n_samples

    def _update_plan_inner(self, plan, city_graph):
        # use sampling rollout
        return sample_from_model(self.model, city_graph, self.n_routes, 
                                 self.min_route_len, self.max_route_len, 
                                 self.cost_obj, self.n_samples)
    

class BCOPlanner(Planner):
    def _update_plan_inner(self, plan, city_graph):
        return bee_colony(city_graph, self.n_routes, self.min_route_len,
                          self.max_route_len, self.cost_obj, 
                          init_scenario=plan, silent=True, 
                          **self.cfg.online_planner.bco)[0]
    
    def get_update_size(self):
        return self.cfg.online_planner.bco.n_iterations
    
    def set_update_size(self, n_iterations):
        self.cfg.online_planner.bco.n_iterations = n_iterations


class NeuralBCOPlanner(Planner):
    def __init__(self, model, cfg, cost_obj):
        super().__init__(cfg, cost_obj)
        self.model = model

    def _update_plan_inner(self, plan, city_graph):
        return bee_colony(city_graph, self.n_routes, self.min_route_len,
                          self.max_route_len, self.cost_obj, 
                          init_scenario=plan, bee_model=self.model,
                          silent=True, **self.cfg.online_planner.neural_bco)[0]

    def get_update_size(self):
        return self.cfg.online_planner.neural_bco.n_iterations

    def set_update_size(self, n_iterations):
        self.cfg.online_planner.neural_bco.n_iterations = n_iterations


def set_iters_from_time_limit(time_limit, planner, data_batch):
    size = planner.get_update_size()
    # construct initial plan
    plan, _ = planner.update_plan(None, data_batch)
    while True:
        log.info(f'trying update size {planner.get_update_size()}')
        _, duration = planner.update_plan(plan, data_batch)
        time_per_iter = duration / size
        if duration < time_limit:
            # estimate how much more size we can add
            delta = time_limit - duration
            if delta < time_per_iter:
                # we're too close to add any, so we're done
                break
           
            else:
                # add the amount we think we can, and repeat
                size += int(delta // time_per_iter)

        elif size == 1:
            # we're as small as we can get, so we're done
            log.warning("Could not find a small-enough update size!")
            break
        else:
            # we're over the time limit, so remove some size.
            delta = duration - time_limit
            size_delta = int(math.ceil(delta / time_per_iter))
            size = max(1, size - size_delta)
        
        # set the planner's update size to the new value

        planner.set_update_size(size)


@hydra.main(version_base=None, config_path="../cfg", config_name="online")
def main(cfg: DictConfig):
    cfg.experiment['logdir'] = None
    device, run_name, _, cost_obj, model = \
        lrnu.process_standard_experiment_cfg(cfg, 'online_')
    model.eval()

    torch.set_grad_enabled(False)

    # instantiate the citygraph
    city_graphs = get_dataset_from_config(cfg.eval.dataset)
    data_batch = Batch.from_data_list(city_graphs)
    if device.type != 'cpu':
        data_batch = data_batch.cuda()
    base_demand = data_batch.demand.clone()

    dynamic_demands = []

    # instantiate the planner
    if 'neural' in cfg.online_planner:
        planner = NeuralConstructionPlanner(model, cfg, cost_obj)
    elif 'bco' in cfg.online_planner:
        planner = BCOPlanner(cfg, cost_obj)
    elif 'neural_bco' in cfg.online_planner:
        planner = NeuralBCOPlanner(model, cfg, cost_obj)

    if cfg.get('dynamic_step_size', False):
        set_iters_from_time_limit(cfg.iter_time_limit_s, planner, data_batch)
        log.info(f"Planning update size: {planner.get_update_size()}")

    all_metrics = None
    plan = None
    all_plans = []
    for _ in tqdm(range(cfg.n_iterations)):
        # TODO update the demand in the citygraph
        # sample gaussian noise with a given width
        dyn_demand = torch.randn(data_batch.demand.shape, device=device)
        dyn_demand *= cfg.demand_stddev
        dynamic_demands.append(dyn_demand)
        if len(dynamic_demands) > cfg.demand_duration_steps:
            # keep only the last n dynamic demands
            dynamic_demands.pop(0)
            
        # sum the last n dynamic demands and add them to the base demand
        step_dyn_demand = torch.stack(dynamic_demands, dim=0).sum(dim=0)
        data_batch.demand = (base_demand + step_dyn_demand).clip(min=0)

        # update the plan
        plan, duration = planner.update_plan(plan, data_batch)
        all_plans.append(plan)
        # simulate the new plan
        result = cost_obj(plan, data_batch)
        metrics = result.get_metrics()
        metrics['step duration'] = torch.tensor([duration])

        if all_metrics is None:
            all_metrics = metrics
        else:
            for key, val in metrics.items():
                all_metrics[key] = torch.cat((all_metrics[key], val))

    # compute a metric of quality in the results, and return them
    metric_stats = {kk: (vv.mean().item(), vv.std().item()) 
                    for kk, vv in all_metrics.items()}
    if cfg.eval.csv:
        # print the results as a CSV row
        csv_row = ','.join([f"{mm:.3f},{ss:.3f}" 
                            for (mm, ss) in metric_stats.values()])
        update_size = planner.get_update_size()
        csv_row = f"{update_size},{csv_row}"
        print(csv_row)
    else:
        print(f"Results for {run_name}:")
        print(f"Update size: {planner.get_update_size()}")
        for kk, (mm, ss) in metric_stats.items():
            print(f"{kk}: {mm}, {ss}")

    # save the routes to a pickle file
    lrnu.dump_routes(run_name, all_plans)


if __name__ == "__main__":
    main()
