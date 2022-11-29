# Energy-optimal Federated Learning

This repository contains the prototypes of five different scheduling algorithms able to minimize the energy consumed on heterogeneous devices during a Federated Learning training round.
The algorithms are described in a preprint available in [HAL](hhttps://hal.archives-ouvertes.fr/hal-0377549) and [arXiv](https://arxiv.org/abs/2209.0621).
This repository was built based on a [previous codebase](https://github.com/llpilla/olar-federated-learning) used to simulate scheduling algorithms that try to minimize the duration of training rounds.

The Python 3 and Bash scripts can be used to test scheduling algorithms in different scenarios with variations in the number of tasks, resources, kinds of resources (cost functions), lower and upper limits of tasks per resources, etc.

If you wish to reproduce all the results in "*Scheduling Algorithms for Federated Learning with Minimal Energy Consumption*", execute `./run_all_total_cost_experiments.sh` to get total cost results and `./run_all_timing_experiments.sh` to get execution time results.
If you want to reproduce the analysis in said manuscript, check [stored\_results](stored\_results) and run `./run_all_analysis.sh`.

## Dependencies

We use numpy, matplotlib, pandas, seaborn, and scipy.
Please run `pip3 install -r requirements.txt` to install them, if needed.

## How to use

If you want to reproduce total cost results only, run `./run_all_total_cost_experiments.py`. It takes about 11 core-hours to run.
Given that the experiments are independent, they could be parallelized with mpi4py.

If you want to reproduce scheduling time results only, run `./run_all_timing_experiments.py`. Beware that they may take several hours to run (about 10 hours in a conventional laptop), and the computer should be left alone while running these experiments to avoid adding unwanted noise to the results.

If you have reproduced these results and want to analyze them, run `./run_analysis_on_new_results.sh`.

If you want to build new experiments, check the several `experiment` and `timing` files available.
