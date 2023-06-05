#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  8 12:47:36 2023

@author: cameronthieme
"""

# Section 1: parameters we play with

# choose number of subdivisions 
phase_subdiv = 18
print('Using %s subdivisions' %phase_subdiv)

# choose training sizes for GP
sample_size = 500
print('Using %s training size' %sample_size)

# number of trials per parameter value
trials = 50

# choose length of trajectories used
# 1 means we sample uniformly in domain
traj_len = 1

# Choose to do measurement error ('meas_err') or step error ('step_err')
noise_type = 'meas_err'
print(noise_type)

# number of restarts for GP optimizer
n_restarts = 29
# choose RBF parameter for GP regression
tau = 6.445
beta = 1

# define boundaries of the problem
lower_bounds = [-0.001, -0.001]
upper_bounds = [90.0, 70.0]

# noise value
noise_std = 0.1

# Section 2: Importing packages and determining parameter values

import CMGDB
import math
import time

import csv
import datetime
import sys

from leslieFunctions import *

# timing everything 
startTime = time.time() # for computing runtime 
latestTime = startTime

# parameter range from Database Paper,  and # boxes in each dim
th1max = 37
th1min = 8
N1 = 40
th2max = 50
th2min = 3
N2 = 40

# finding correct row/column in grid
job = int(sys.argv[1]) # Input file name
# job = 2
xIter = job % N1
yIter = math.floor(job/N1) % N2

# theta value
th1 = xIter * (th1max - th1min) / N1 + (th1max - th1min) / (2 * N1) + th1min
th2 = yIter * (th2max - th2min) / N2 + (th2max - th2min) / (2 * N2) + th2min
print('th1: %s' %th1)
print('th2: %s' %th2)

# Section 3: Getting True Conley-Morse Info

# names for saving files
curDate = datetime.datetime.now() # will use date and time to distinguish files
fileName = '_th1' + str(th1) + 'th2' + str(th2) + 'job' + str(job) +  curDate.strftime("%b") + '_' + curDate.strftime('%d') + 'at'+ curDate.strftime('%I') + curDate.strftime('%M') 
fileNameMGtrue = 'MorseGraphs/MG_True' + fileName

modelTrue = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, LeslieIntervalBox)
morse_graph_True, map_graph_True = CMGDB.ComputeConleyMorseGraph(modelTrue)
mgTrue = CMGDB.PlotMorseGraph(morse_graph_True)
mgTrue.save(fileNameMGtrue)


for trial in range(trials):
    # Sampling training data
    x_train, y_train0, y_train1 = sampGenLes(
        noise_std = noise_std,
        sample_size = sample_size,
        noise_type = noise_type,
        traj_len = traj_len,
        th1 = th1,
        th2 = th2,
        lower_bounds = lower_bounds,
        upper_bounds = upper_bounds)
    
    # Train a GP with the data above
    gp0 = GP(X_train = x_train,
             Y_train = y_train0,
             tau = tau,
             beta = beta,
             noise_std = noise_std,
             n_restarts = n_restarts)
    gp1 = GP(X_train = x_train,
             Y_train = y_train1,
             tau = tau,
             beta = beta,
             noise_std = noise_std,
             n_restarts = n_restarts)
    
    # check if map becomes zero map
    zero_counter = zero_checker(gp0 = gp0,
                                gp1 = gp1,
                                lower_bounds = lower_bounds,
                                upper_bounds = upper_bounds)
    
    # Section 4: Running CMGDB & Checking Isomorphism
    
    curDate = datetime.datetime.now() # will use date and time to distinguish files
    
    if zero_counter == 0: # only triggers if mean nonzero
        
        # 4(a): running CMGDB
        # this is most time-consuming part by far (if we are using reasonable subdivs)
        #CMGDB computations
        
        # defining model parameters
        def GP_Box(rect):
            return BoxMapSD(rect, gp0, gp1)
        modelGP = CMGDB.Model(phase_subdiv, lower_bounds, upper_bounds, GP_Box)
        
        # running the hard computation
        morse_graph_GP, map_graph_GP = CMGDB.ComputeConleyMorseGraph(modelGP)
        
        # saving Morse Graphs
        fileNameMGgp = 'MorseGraphs/MG_GP_trial' + str(trial) + fileName
        mgGP = CMGDB.PlotMorseGraph(morse_graph_GP)
        mgGP.save(fileNameMGgp)
        
        # Section 4b: Checking For Nontrivial Graph Isomorphisms
        
        identical = ConleyMorseMatcher(morse_graph_True, morse_graph_GP)
    
    else:
        identical = 0
    
    # Section 5: Saving Data
    
    # Naming File
    fileNameResults = 'ResultTables/results_trial' + str(trial)  + fileName + '.csv'

    # Info we want to save
    infoList = [th1, th2, identical, zero_counter]
    
    # Writing info to CSV
    with open(fileNameResults, mode='w', newline='') as file:
        # Create a writer object
        writer = csv.writer(file)
        # Write the list to the CSV file
        writer.writerow(infoList)

# Recording Execution Times
executionTime = time.time() - startTime
print('Total execution Time:')
print('Execution time, in seconds: ' + str(executionTime))
print('Execution time, in minutes: ' + str(executionTime/60))










