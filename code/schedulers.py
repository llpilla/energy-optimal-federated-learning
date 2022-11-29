"""
Module containing scheduling algorithms.
"""

import numpy as np
import heapq

def mc2mkp(
        tasks,
        resources,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources based on the dynamic
    programming algorithm for the (MC)^2MKP problem.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    # K = minimal costs
    # I = Partial solutions (schedule for a given resource and t)
    K = np.full(shape=(resources, tasks+1), fill_value=np.inf)
    I = np.zeros(shape=(resources, tasks+1), dtype=int)
    # Solutions for Z_1
    for j in range(lower_limit[0], upper_limit[0]+1):
        K[0][j] = cost[0][j]
        I[0][j] = j
    # Solutions for Z_i
    for i in range(1, resources):
        # All possible values for x_i
        for j in range(lower_limit[i], upper_limit[i]+1):
            c = cost[i][j]
            for t in range(j, tasks+1):
                if K[i-1][t-j] + c < K[i][t]:
                    # New best solution for Z_i(t)
                    K[i][t] = K[i-1][t-j] + c
                    I[i][t] = j
    # Gets the final assignment from the support matrices
    assignment = np.zeros(resources, dtype=int)
    t = tasks
    for i in reversed(range(resources)):
        j = I[i][t]  # Number of tasks to resource i
        assignment[i] = j
        t = t - j    # index for the solution for resource i-1
    return assignment


def marin(
        tasks,
        resources,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources using MarIn.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources

    Notes
    -----
    Based on OLAR, published on “Optimal task assignment for
    heterogeneous federated learning devices,” in 2021 IEEE
    International Parallel and Distributed Processing Symposium
    (IPDPS), 2021, pp. 661–670.
    """
    # Initialization
    heap = []
    # Assigns lower limit to all resources
    assignment = np.copy(lower_limit)
    for i in range(resources):
        # Initializes the heap
        if assignment[i] < upper_limit[i]:
            heap.append(((cost[i][assignment[i]+1]
                          - cost[i][assignment[i]]), i))
    heapq.heapify(heap)
    # Computes zeta (sum of lower limits)
    zeta = np.sum(lower_limit)
    # Iterates assigning the remaining tasks
    for t in range(zeta+1, tasks+1):
        c, j = heapq.heappop(heap)  # Find minimum cost
        assignment[j] += 1  # Assigns task t
        # Checks if more tasks can be assigned to j
        if assignment[j] < upper_limit[j]:
            heapq.heappush(heap, ((cost[j][assignment[j]+1]
                                   - cost[j][assignment[j]]), j))
    return assignment


def marco(
        tasks,
        resources,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources using MarCo.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    heap = []
    # Assigns lower limit to all resources
    assignment = np.copy(lower_limit)
    for i in range(resources):
        # Initializes the heap
        if assignment[i] < upper_limit[i]:
            heap.append((cost[i][1] - cost[i][0], i))
    heapq.heapify(heap)
    # Computes how many tasks have been assigned already
    t = np.sum(lower_limit)
    # Iterates assigning the remaining tasks in groups
    while t < tasks:
        c, j = heapq.heappop(heap)  # Find minimum cost
        # Finds how many tasks the resource can still receive
        a = min(upper_limit[j] - lower_limit[j], tasks - t)
        assignment[j] += a  # Assigns a group of tasks
        t += a
    return assignment


def mardecun(
        tasks,
        resources,
        cost,
        lower_limit
        ):
    """
    Finds an assignment of tasks to resources using MarDecUn.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Assigns lower limit to all resources
    assignment = np.copy(lower_limit)
    # Computes how many tasks have been assigned already
    l = np.sum(lower_limit)
    t = tasks - l  # and how many are still left to assign
    # Resource with minimal cost for the remaining tasks
    min_resource = 0
    min_cost = cost[0][lower_limit[0] + t] - cost[0][lower_limit[0]]
    for i in range(1, resources):
        new_cost = cost[i][lower_limit[i] + t] - cost[i][lower_limit[i]]
        if min_cost > new_cost:
            # New minimal
            min_resource = i
            min_cost = new_cost
    # Assigns all remaining tasks to the same resource
    assignment[min_resource] += t
    return assignment

def mardec(
        tasks,
        resources,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Finds an assignment of tasks to resources using MarDec.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Initialization
    # Assigns lower limit to all resources
    assignment = np.copy(lower_limit)
    # Computes how many tasks have been assigned already
    l = np.sum(lower_limit)
    tasks_left = tasks - l
    # List of resources
    r = np.arange(resources)
    # Resources with upper limits
    Rlim = r[upper_limit - lower_limit < tasks_left]
    # Resources without upper limits
    Runl = r[upper_limit - lower_limit >= tasks_left]
    total_cost = np.inf       # No valid solutions to start

    # Case 1: solution with a resource from Runl at intermediary capacity
    if Runl.size > 0:
        # Prepares and gets the matrices in one single function
        K, I = mcmkp_matrices(tasks_left, Rlim, cost, lower_limit, upper_limit)
        # Evaluates all partial solutions
        for t in range(0, tasks_left+1):
            # Finds the unlimited resource with the smallest cost
            # when receiving t extra tasks
            min_resource = Runl[0]
            min_cost = cost[min_resource][lower_limit[min_resource] + t] \
                       - cost[min_resource][lower_limit[min_resource]]
            for i in range(1, Runl.size):
                new_resource = Runl[i]
                new_cost = cost[new_resource][lower_limit[new_resource] + t] \
                           - cost[new_resource][lower_limit[new_resource]]
                if new_cost < min_cost:
                    min_cost = new_cost
                    min_resource = new_resource
            # Checks if it finds a better solution with this resource
            if min_cost + K[Rlim.size - 1][tasks_left - t] < total_cost:
                # Updates the best solution
                total_cost = min_cost + K[Rlim.size - 1][tasks_left - t]
                assignment = translate(I, lower_limit, tasks_left - t, Rlim)
                assignment[min_resource] += t

    # Case 2: solution with a resource from Rlim at intermediary capacity
    for i in range(Rlim.size):
        # Remove the i-th limited resource for evaluation
        Reval = np.delete(Rlim, i)
        # Prepares and gets the matrices in one single function
        K, I = mcmkp_matrices(tasks_left, Reval, cost, lower_limit, upper_limit)
        # Evaluates all partial solutions
        max_tasks = min(tasks_left+1, upper_limit[Rlim[i]] - lower_limit[Rlim[i]])
        for t in range(0, max_tasks):
            # Finds the cost for the limited resource of interest
            # when receiving t extra tasks
            min_resource = Rlim[i]
            min_cost = cost[min_resource][lower_limit[min_resource] + t] \
                       - cost[min_resource][lower_limit[min_resource]]
            # Checks if it finds a better solution with this resource
            if min_cost + K[Reval.size - 1][tasks_left - t] < total_cost:
                # Updates the best solution
                total_cost = min_cost + K[Reval.size - 1][tasks_left - t]
                assignment = translate(I, lower_limit, tasks_left - t, Reval)
                assignment[min_resource] += t

    # Returns the best schedule found
    return assignment

def mcmkp_matrices(
        tasks,
        R,
        cost,
        lower_limit,
        upper_limit
        ):
    """
    Runs a simple MCMKP algorithm and returns the support matrices.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    R : np.ndarray(dtype=int)
        List of resources to consider
    cost : np.ndarray(shape=(resources, tasks+1))
        Cost functions per resource (C)
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    upper_limit : np.array(shape=(resources), dtype=int)
        Upper limit of number of tasks per resource

    Returns
    -------
    np.array(shape=(R.size, tasks+1))
        Minimal costs found
    np.array(shape=(R.size, tasks+1))
        Partial solutions

    Notes
    -----
    The proposed solutions consider only two scenarios:
        - Mapping zero extra tasks to a resource (its lower limit)
        - Mapping the most tasks possible to a resource (its upper limit)
    """
    # Initialization
    resources = R.size
    # K = minimal costs
    # I = Partial solutions (schedule for a given resource and t)
    K = np.full(shape=(resources, tasks+1), fill_value=np.inf)
    I = np.zeros(shape=(resources, tasks+1), dtype=int)
    # Solutions for Z_1
    # Assigning zero extra tasks to the first resource
    K[0][0] = 0
    I[0][0] = 0
    # Assigning (upper limit - lower limit) extra tasks to the first resource
    k = R[0]  # first resource in the list of interest
    j = upper_limit[k] - lower_limit[k]  # number of extra tasks
    c = cost[k][upper_limit[k]] - cost[k][lower_limit[k]]  # difference in cost
    K[0][j] = c
    I[0][j] = j
    # Solutions for Z_i
    for i in range(1, resources):
        # Solutions mapping zero extra tasks to the resource
        K[i] = K[i-1]
        # Solutions mapping the most extra tasks to the resource
        k = R[i]  # i resource in the list
        j = upper_limit[k] - lower_limit[k]  # number of extra tasks
        c = cost[k][upper_limit[k]] - cost[k][lower_limit[k]]  # cost
        for t in range(j, tasks+1):
            if K[i-1][t-j] + c < K[i][t]:
                # New best solution for Z_i(t)
                K[i][t] = K[i-1][t-j] + c
                I[i][t] = j
    return K, I


def translate(
        I,
        lower_limit,
        tasks,
        R
        ):
    """
    Translates a partial MCMKP solution to a schedule.

    Parameters
    ----------
    I : np.array(shape=(R.size, tasks+1))
        Partial solutions
    lower_limit : np.array(shape=(resources), dtype=int)
        Lower limit of number of tasks per resource
    tasks : int
        Number of tasks (tau)
    R : np.ndarray(dtype=int)
        List of resources to consider

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources
    """
    # Assigns the lower limits to all resources
    assignment = np.copy(lower_limit)
    # Goes through the partial solutions to find the extra tasks to assign
    t = tasks
    for i in reversed(range(R.size)):
        j = I[i][t]  # Number of extra tasks to the i-th resource
        resource = R[i]
        assignment[resource] += j
        t = t - j    # index for the solution for the i-th -1 resource
    return assignment


def fedavg(
        tasks,
        resources,
        ):
    """
    Finds an assignment of tasks to resources using FedAvg.

    Parameters
    ----------
    tasks : int
        Number of tasks (tau)
    resources : int
        Number of resources (R)

    Returns
    -------
    np.array(shape=(resources))
        Assignment of tasks to resources

    Notes
    -----
    FederatedAveraging is presented in
    "McMahan, Brendan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise
    Aguera y Arcas. "Communication-efficient learning of deep networks from
    decentralized data." In Artificial Intelligence and Statistics,
    pp. 1273-1282. PMLR, 2017."

    The algorithm is cost-oblivious and splits the tasks equality among
    the resources.
    """
    # divides the tasks as equally as possible
    mean_tasks = tasks // resources
    # but it sill may have some leftovers
    leftover = tasks % resources
    assignment = np.full(shape=resources, fill_value=mean_tasks)
    if leftover > 0:
        # adds the leftover to the first resources
        assignment[0:leftover] += 1
    return assignment
