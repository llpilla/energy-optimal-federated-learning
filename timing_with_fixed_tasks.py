"""# Description of the experiment:
#
# - We generate the costs to up to 2.000 tasks for 20 to 80 resources
#   with steps of 10.
# - All costs follow linear functions (i.e., constant marginal costs)
#   with RNG seeds [0..79].
# - We schedule 2.000 tasks.
# - We run (MC)^2MKP, MarIn, MarCo, MarDec, MarDecUn, and FedAvg.
# - All resources have a lower limit of 1.
# - The first half of the resources have no upper limit.
# - The second half has an upper limit of 2*(tasks/resources).
# - Each sample is composed of 5 executions of the schedulers.
# - We get 20 samples for each pair (scheduler, tasks)
# - The order of execution of the different schedulers is
#   randomly defined. We set an initial RNG seed = 1000 and increase
#   it every time we need a new order.
"""


import timeit
import numpy as np
import code.support as support

# File containing the results
logger = support.Logger('results_of_timing_with_fixed_tasks.csv')
tasks = 2000
min_resources = 20
max_resources = 81
step_resources = 20
size_of_sample = 5
number_of_samples = 20
shuffle_initial_seed = 1000
scheduler_name = ['(MC)2MKP', 'MarIn', 'MarCo', 'MarDecUn', 'MarDec', 'FedAvg']


def run_timing():
    # Stores the description of the experiments
    logger.header(__doc__)
    # Header of the CSV file
    logger.store('Scheduler,Tasks,Resources,Time')
    # Runs experiments for 10000 tasks
    run_for_fixed_tasks()
    # Finishes logging
    logger.finish()


def run_for_fixed_tasks():
    """
    Runs experiments for a fixed number of tasks.
    """

    # counting the rounds of the experiment to update the RNG seed
    rounds = 0
    # runs experiments for all numbers of resources
    for resources in range(min_resources, max_resources, step_resources):
        print(f'- Running experiments with {resources} resources')
        # sets the string to be used for each scheduler
        calls = []
        # version that verifies the results
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.mc2mkp(tasks, {resources}, cost, lower_limit, upper_limit)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.marin(tasks, {resources}, cost, lower_limit, upper_limit)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.marco(tasks, {resources}, cost, lower_limit, upper_limit)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.mardecun(tasks, {resources}, cost, lower_limit)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.mardec(tasks, {resources}, cost, lower_limit, upper_limit)))")
        #calls.append(f"print(support.check_total_assigned(tasks, schedulers.fedavg(tasks, {resources})))")
        # version that does not verify the results
        calls.append(f"a = schedulers.mc2mkp(tasks, {resources}, cost, lower_limit, upper_limit)")
        calls.append(f"a = schedulers.marin(tasks, {resources}, cost, lower_limit, upper_limit)")
        calls.append(f"a = schedulers.marco(tasks, {resources}, cost, lower_limit, upper_limit)")
        calls.append(f"a = schedulers.mardecun(tasks, {resources}, cost, lower_limit)")
        calls.append(f"a = schedulers.mardec(tasks, {resources}, cost, lower_limit, upper_limit)")
        calls.append(f"a = schedulers.fedavg(tasks, {resources})")

        # Setup up to generate 80 resources with 2001 costs
        setup = f"""
import numpy as np
import code.schedulers as schedulers
import code.devices as devices
import code.support as support

rng_seed_resources = 0
tasks = 2000
seed_for_random = 0
resources = {resources}
k = 1
# Initializes the cost matrix with zeros
cost = np.zeros(shape=(resources, tasks+1))
# Fills the cost matrix with costs based on a linear function
base_seed = rng_seed_resources
for i in range(resources):
    devices.create_linear_costs(base_seed, cost, i, tasks)
    base_seed += 1
# Prepares the upper and lower limit arrays
lower_limit = np.full(shape=resources, fill_value=1, dtype=int)
upper_limit = np.full(shape=resources, fill_value={tasks}, dtype=int)
np.put(upper_limit, np.arange(resources//2, resources), 2*(tasks//resources))
"""

        # gathers all samples for a given (number of tasks, scheduler)
        for sample in range(number_of_samples):
            # sets the RNG seed and generates an order of execution
            np.random.seed(shuffle_initial_seed + rounds)
            rounds += 1
            order = np.arange(6)  # six different schedulers
            np.random.shuffle(order)  # random order

            # gathers samples for all schedulers
            for i in order:
                # gathers one sample
                timing = timeit.timeit(setup=setup,
                                       stmt=calls[i],
                                       number=size_of_sample)
                # stores the timing information
                logger.store(f'{scheduler_name[i]},{tasks},{resources},{timing}')


if __name__ == '__main__':
    run_timing()
