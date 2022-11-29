#!/usr/bin/env python
# coding: utf-8

# # Analysis of the experiments with different schedulers for Federated Learning - Total cost

# In[ ]:


# modules for the analysis
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_theme(style="whitegrid")


# In[ ]:


# setting colors
c_mardec = [sns.color_palette("GnBu", 10)[8]]
c_fedavg = [sns.color_palette("YlOrBr", 10)[8]]
c_marin = [sns.color_palette("colorblind", 10)[4]]
c_marco = [sns.color_palette("PuOr", 10)[8]]
c_mc2mkp = [sns.color_palette("BuGn", 10)[8]]

sns.set_palette(c_mc2mkp + c_marin + c_marco + c_mardec + c_fedavg)

schedulers = ['(MC)2MKP', 'MarIn', 'MarCo', 'MarDec', 'FedAvg']


# ## Results with random costs

# In[ ]:


# reads the result file
results = pd.read_csv('results_with_random_costs.csv', comment='#')
print('- Results with random costs')
results.head(9)


# In[ ]:


# checking the number of results versus the expected number of results
expected_number = 41*5*2  # 41 numbers of tasks, 5 schedulers, 2 numbers of resources
print(f'-- Number of results: {len(results)} (expected: {expected_number})')


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
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)

ax = sns.lineplot(data=results[results.Resources == 10],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-random-10.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-random-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of (MC)^2MKP
mc2mkp_cost = results[results['Scheduler'] == '(MC)2MKP']['Total Cost'].reset_index(drop=True)
for scheduler in schedulers[1:]:
    other_cost = results[results['Scheduler'] == scheduler]['Total Cost'].reset_index(drop=True)
    greater = np.sum(other_cost > mc2mkp_cost)
    equal = np.sum(other_cost == mc2mkp_cost)
    less = np.sum(other_cost < mc2mkp_cost)
    print(f'Number of times {scheduler} provides a Total Cost that is greater, equal, or smaller than (MC)^2MKP: ' +
          f'{greater}, {equal}, {less}.')


# ---
# ## Results with increasing marginal costs

# In[ ]:


# reads the result file
results = pd.read_csv('results_with_increasing_marginal_costs.csv', comment='#')
print('- Results with increasing marginal costs')
results.head(9)


# In[ ]:


print(f'-- Number of results: {len(results)} (expected: {expected_number})')


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
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 160000)

ax = sns.lineplot(data=results[results.Resources == 10],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-increasing-10.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 120000)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-increasing-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of (MC)^2MKP
mc2mkp_cost = results[results['Scheduler'] == '(MC)2MKP']['Total Cost'].reset_index(drop=True)
for scheduler in schedulers[1:]:
    other_cost = results[results['Scheduler'] == scheduler]['Total Cost'].reset_index(drop=True)
    greater = np.sum(other_cost > mc2mkp_cost)
    equal = np.sum(other_cost == mc2mkp_cost)
    less = np.sum(other_cost < mc2mkp_cost)
    print(f'Number of times {scheduler} provides a Total Cost that is greater, equal, or smaller than (MC)^2MKP: ' +
          f'{greater}, {equal}, {less}.')


# ---
# ## Results with constant marginal costs

# In[ ]:


# reads the result file
results = pd.read_csv('results_with_constant_marginal_costs.csv', comment='#')
print('- Results with constant marginal costs')
results.head(9)


# In[ ]:


print(f'-- Number of results: {len(results)} (expected: {expected_number})')


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
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 30000)

ax = sns.lineplot(data=results[results.Resources == 10],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-constant-10.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 30000)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-constant-100.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of (MC)^2MKP
mc2mkp_cost = results[results['Scheduler'] == '(MC)2MKP']['Total Cost'].reset_index(drop=True)
for scheduler in schedulers[1:]:
    other_cost = results[results['Scheduler'] == scheduler]['Total Cost'].reset_index(drop=True)
    greater = np.sum(other_cost > mc2mkp_cost)
    equal = np.sum(other_cost == mc2mkp_cost)
    less = np.sum(other_cost < mc2mkp_cost)
    print(f'Number of times {scheduler} provides a Total Cost that is greater, equal, or smaller than (MC)^2MKP: ' +
          f'{greater}, {equal}, {less}.')


# ---
# ## Results with decreasing marginal costs

# In[ ]:


# reads the result file
results = pd.read_csv('results_with_decreasing_marginal_costs.csv', comment='#')
print('- Results with decreasing marginal costs')
results.head(9)


# In[ ]:


print(f'-- Number of results: {len(results)} (expected: {expected_number})')


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
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 500)

ax = sns.lineplot(data=results[results.Resources == 10],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-decreasing-10.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 3000)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-decreasing-100.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(1465, 1472)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-decreasing-100-zoom.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of (MC)^2MKP
mc2mkp_cost = results[results['Scheduler'] == '(MC)2MKP']['Total Cost'].reset_index(drop=True)
for scheduler in schedulers[1:]:
    other_cost = results[results['Scheduler'] == scheduler]['Total Cost'].reset_index(drop=True)
    greater = np.sum(other_cost > mc2mkp_cost)
    equal = np.sum(other_cost == mc2mkp_cost)
    less = np.sum(other_cost < mc2mkp_cost)
    print(f'Number of times {scheduler} provides a Total Cost that is greater, equal, or smaller than (MC)^2MKP: ' +
          f'{greater}, {equal}, {less}.')


# ---
# ## Results with constant marginal costs and no upper limits

# In[ ]:


# reads the result file
results = pd.read_csv('results_with_constant_marginal_costs_no_upper_limit.csv', comment='#')
print('- Results with constant marginal costs and no upper limits')
results.head(9)


# In[ ]:


print(f'-- Number of results: {len(results)} (expected: {expected_number})')


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
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 30000)

ax = sns.lineplot(data=results[results.Resources == 10],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-constant-10-no-u.pdf", bbox_inches='tight')


# In[ ]:


# Sets figure parameters
plt.figure(figsize=(6,5))
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 13
plt.rcParams['ytick.labelsize'] = 13
plt.rcParams['legend.fontsize'] = 12
plt.xlabel('Number of tasks (T)', fontsize=13)
plt.ylabel('Total cost (a.u.)', fontsize=13)
plt.xticks(rotation=15)
plt.ylim(0, 30000)

ax = sns.lineplot(data=results[results.Resources == 100],
                  x='Tasks',
                  y='Total Cost',
                  hue='Scheduler',
                  style='Scheduler',
                  dashes=False,
                  markers=True,
                  linewidth=2,
                  markersize=8)

sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=True)

plt.savefig("fig-constant-100-no-u.pdf", bbox_inches='tight')


# In[ ]:


# Checking how many times other schedulers meet the performance of (MC)^2MKP
mc2mkp_cost = results[results['Scheduler'] == '(MC)2MKP']['Total Cost'].reset_index(drop=True)
for scheduler in ['MarIn', 'MarCo', 'MarDecUn', 'FedAvg']:
    other_cost = results[results['Scheduler'] == scheduler]['Total Cost'].reset_index(drop=True)
    greater = np.sum(other_cost > mc2mkp_cost)
    equal = np.sum(other_cost == mc2mkp_cost)
    less = np.sum(other_cost < mc2mkp_cost)
    print(f'Number of times {scheduler} provides a Total Cost that is greater, equal, or smaller than (MC)^2MKP: ' +
          f'{greater}, {equal}, {less}.')

