python learning/eval_saved_routes.py +eval=lavalDA +init=load \
    init.path=stl_da_routes.pkl eval.n_routes=79 experiment/cost_function=pp \
    hydra/job_logging=disabled \
    +experiment.cost_function.kwargs.symmetric_routes=false > \
    result_csvs/post_simfix/stl_laval_1.0.csv