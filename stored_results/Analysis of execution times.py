#!/usr/bin/env python
# coding: utf-8

# # Analysis of the experiments with different schedulers for Federated Learning - Execution time

# In[ ]:


# modules for the analysis
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

sns.set_theme(style="whitegrid")


# In[ ]:


# setting colors
c_mardec = [sns.color_palette("GnBu", 10)[8]]
c_mardecun = [sns.color_palette("GnBu", 10)[4]]
c_fedavg = [sns.color_palette("YlOrBr", 10)[8]]
c_marin = [sns.color_palette("colorblind", 10)[4]]
c_marco = [sns.color_palette("PuOr", 10)[8]]
c_mc2mkp = [sns.color_palette("BuGn", 10)[8]]

sns.set_palette(c_mc2mkp + c_marin + c_marco + c_mardecun + c_mardec + c_fedavg)

schedulers = ['(MC)2MKP', 'MarIn', 'MarCo', 'MarDecUn', 'MarDec', 'FedAvg']


# ## Results with increasing numbers of tasks

# In[ ]:


# reads the result file
results = pd.read_csv('results_of_timing_with_fixed_resources.csv', comment='#')
print('- Results with increasing numbers of tasks')
results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
expected_number = 10*6*1*20  # 10 numbers of tasks, 6 schedulers, 1 number of resources, 20 samples
print(f'-- Number of results: {len(results)} (expected: {expected_number})')


# In[ ]:


# Transforming the values of all samples
# Each sample contains the time of 5 repetitions
# This transformation gives the average time of the 5 repetitions
# and converts from seconds to us
results['avg'] = results['Time']*1000000/5


# In[ ]:


print('-- Generating figures')

# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Execution time (us, log scale)', fontsize=13)
plt.xticks(range(200,2001,200))
plt.xticks(rotation=15)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='avg',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8,
                  hue_order=schedulers)

plt.ylim(1, 100000000)
plt.yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 100000000])
plt.yscale('log')

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-time-fixed-resources.pdf", bbox_inches='tight')


# ### Description of results

# In[ ]:


print('- Result description')
for sched in schedulers:
    for tasks in (200, 2000):
        res = results[(results['Scheduler'] == sched) &
                      (results['Tasks'] == tasks)].avg
        print(f'Scheduler {sched} with {tasks} tasks')
        print(res.describe())
        print('\n')


# ### Distributions of the results

# In[ ]:


print('-- Checking the distribution of origin for different results')
# Checking all schedulers
np.random.seed(2022)
for sched in schedulers:
    print(f'\nResuls for scheduler {sched}:')
    for tasks in range(200,2001,200):
        res = list(results[(results['Scheduler'] == sched) &
                           (results['Tasks'] == tasks)].avg)
        print(f'- {tasks} tasks')
        print(stats.kstest(res, 'norm', args=(np.mean(res), np.std(res))))


# In[ ]:


# Statistical comparison between MarCo and MarDecUn
# Using Mann-Whitney U test as results do not follow normal distributions some times (p-values < 0.05)
for tasks in range(200,2001,200):
    marco = list(results[(results['Scheduler'] == 'MarCo') &
                         (results['Tasks'] == tasks)].avg)
    mardec = list(results[(results['Scheduler'] == 'MarDec') &
                         (results['Tasks'] == tasks)].avg)
    print(f'Mann-Whitney U test - ({tasks} tasks).')
    print(stats.mannwhitneyu(marco, mardec, alternative='two-sided'))


# MarCo and MarDec perform differently (p-values < 0.05).

# ## Results with increasing numbers of resources

# In[ ]:


# reads the result file
results = pd.read_csv('results_of_timing_with_fixed_tasks.csv', comment='#')
print('- Results with increasing numbers of resources')
results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
expected_number = 1*6*4*20  # 1 number of tasks, 6 schedulers, 4 numbers of resources, 20 samples
print(f'-- Number of results: {len(results)} (expected: {expected_number})')


# In[ ]:


# Transforming the values of all samples
# Each sample contains the time of 5 repetitions
# This transformation gives the average time of the 5 repetitions
# and converts from seconds to us
results['avg'] = results['Time']*1000000/5


# In[ ]:


print('-- Generating figures')

# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of resources (n)', fontsize=13)
plt.ylabel('Execution time (us, log scale)', fontsize=13)
plt.xticks(range(20,81,20))
plt.xticks(rotation=15)

ax = sns.lineplot(data=results[results.Tasks == 2000],
                  x='Resources',
                  y='avg',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8,
                  hue_order=schedulers)

plt.ylim(1, 100000000)
plt.yticks([1, 10, 100, 1000, 10000, 100000, 1000000, 100000000])
plt.yscale('log')

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-time-fixed-tasks.pdf", bbox_inches='tight')


# ### Description of results

# In[ ]:


print('- Result description')
for sched in schedulers:
    for resources in (20, 80):
        res = results[(results['Scheduler'] == sched) &
                      (results['Resources'] == resources)].avg
        print(f'Scheduler {sched} with {resources} resources')
        print(res.describe())
        print('\n')


# ### Distributions of the results

# In[ ]:


print('-- Checking the distribution of origin for different results')
# Checking all schedulers
np.random.seed(2022)
for sched in schedulers:
    print(f'\nResuls for scheduler {sched}:')
    for resources in range(20,81,20):
        res = list(results[(results['Scheduler'] == sched) &
                           (results['Resources'] == resources)].avg)
        print(f'- {resources} resources')
        print(stats.kstest(res, 'norm', args=(np.mean(res), np.std(res))))


# In[ ]:


# Statistical comparison between MarCo and MarDecUn
# Using Mann-Whitney U test as results do not follow normal distributions some times (p-values < 0.05)
for resources in range(20,81,20):
    marco = list(results[(results['Scheduler'] == 'MarCo') &
                         (results['Resources'] == resources)].avg)
    mardec = list(results[(results['Scheduler'] == 'MarDec') &
                         (results['Resources'] == resources)].avg)
    print(f'Mann-Whitney U test - ({resources} resources).')
    print(stats.mannwhitneyu(marco, mardec, alternative='two-sided'))


# MarCo and MarDec perform differently (p-values < 0.05).
