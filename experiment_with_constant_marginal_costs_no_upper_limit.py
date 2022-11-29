"""# Description of the experiment:
#
# - We generate the costs to up to 5.000 tasks for 10 and 100 resources.
# - All costs follow linear functions (i.e., constant marginal costs)
#   with RNG seeds [500..599].
# - We schedule from 1.000 to 5.000 tasks in increments of 100.
# - We run (MC)^2MKP, MarIn, MarCo, MarDecUn, and FedAvg.
# - All resources have a lower limit of 5 and no upper limit.
# - Every result is verified and logged to a CSV file.
"""

import numpy as np
import code.support as support
import code.schedulers as schedulers
import code.devices as devices


# File containing the results
logger = support.Logger('results_with_constant_marginal_costs_no_upper_limit.csv')
rng_seed_resources = 500
min_tasks = 1000
max_tasks = 5001
step_tasks = 100


def run_constant_marginal_costs():
    # Stores the description of the experiments
    logger.header(__doc__)
    # Header of the CSV file
    logger.store('Scheduler,Tasks,Resources,Total Cost')
    # Runs experiments for 10 resources
    run_for_n_resources(10)
    # Runs experiments for 100 resources
    run_for_n_resources(100)
    # Finishes logging
    logger.finish()


def run_for_n_resources(resources):
    """
    Runs experiments for a number of resources.

    Parameters
    ----------
    resources : int
        Number of resources
    """
    print(f'- Running experiment for {resources} resources.')
    # Initializes the cost matrix with zeros
    cost = np.zeros(shape=(resources, max_tasks+1))
    # Fills the cost matrix with costs based on a linear function
    base_seed = rng_seed_resources
    for i in range(resources):
        devices.create_linear_costs(base_seed, cost, i, max_tasks)
        base_seed += 1

    # Prepares the lower limit array
    lower_limit = np.full(shape=resources, fill_value=5, dtype=int)

    # Iterates over the number of tasks running all schedulers
    for tasks in range(min_tasks, max_tasks, step_tasks):
        if tasks % 1000 == 0:
            print(f'-- Running with {tasks} tasks.')
        # Prepares the upper limit array
        upper_limit = np.full(shape=resources, fill_value=tasks, dtype=int)

        # Runs the algorithms
        a = schedulers.mc2mkp(tasks, resources, cost, lower_limit, upper_limit)
        check_and_store('(MC)2MKP', tasks, resources, a, cost)
        a = schedulers.marin(tasks, resources, cost, lower_limit, upper_limit)
        check_and_store('MarIn', tasks, resources, a, cost)
        a = schedulers.marco(tasks, resources, cost, lower_limit, upper_limit)
        check_and_store('MarCo', tasks, resources, a, cost)
        a = schedulers.mardecun(tasks, resources, cost, lower_limit)
        check_and_store('MarDecUn', tasks, resources, a, cost)
        a = schedulers.fedavg(tasks, resources)
        check_and_store('FedAvg', tasks, resources, a, cost)


def check_and_store(name, tasks, resources, assignment, cost):
    """
    Checks if the results are correct and stores them in the logger.

    Parameters
    ----------
    name : string
        Name of the scheduler
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    assignment : np.array(shape=(resources))
        Assignment of tasks to resources
    cost : np.array(shape=(resources, tasks+1))
        Cost functions per resource (C)
    """
    total_cost = support.get_total_cost(cost, assignment)
    if support.check_total_assigned(tasks, assignment) == False:
        print(f'-- {name} failed to assign {tasks} tasks to' +
              f' {resources} resources ({np.sum(assignment)}' +
              ' were assigned).')
    logger.store(f'{name},{tasks},{resources},{total_cost}')


if __name__ == '__main__':
    run_constant_marginal_costs()
