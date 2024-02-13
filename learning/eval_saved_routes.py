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

import logging as log

from omegaconf import DictConfig
import hydra

from torch_geometric.loader import DataLoader
from simulation.citygraph_dataset import get_dataset_from_config
import learning.utils as lrnu


@hydra.main(version_base=None, config_path="../cfg", 
            config_name="eval_model_mumford")
def main(cfg: DictConfig):
    global DEVICE    
    DEVICE, run_name, _, cost_fn, _ = \
        lrnu.process_standard_experiment_cfg(cfg, weights_required=False)
    
    test_ds = get_dataset_from_config(cfg.eval.dataset)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size)

    # evaluate the model on the dataset
    lrnu.test_method(None, test_dl, cfg.eval, cost_fn, silent=False, 
                     init_solution_file=cfg.routes, device=DEVICE, 
                     return_routes=True)


if __name__ == "__main__":
    main()
