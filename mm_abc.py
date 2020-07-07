# IMPORT PACKAGES
from __future__ import division, print_function
from numpy.linalg import norm, pinv
from numpy.random import choice
from numpy import array, expand_dims, repeat, concatenate, arange, asarray, linspace, max, min, zeros, abs, sum, tan, log, mean, std, dot, squeeze, diag, ones
from math import ceil
from json import load
from time import gmtime, strftime
from six.moves import input
import sys
import csv


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# MICHAELIS-MENTEN KINETICS
def f(args, r, t, theta):
    K_M = theta[:,[0]]
    k_cat = theta[:,[1]]

    P = r[:,[0]]

    fPd = ( (k_cat * args['E_0']) * (args['S_0'] - P) ) / ( K_M + (args['S_0'] - P) )

    return fPd


# RUNGE-KUTTA METHOD
def rk4(args, r, t, theta):
    """ Runge-Kutta 4 method """
    k1 = args["h"]*f(args, r, t, theta)
    k2 = args["h"]*f(args, r+0.5*k1, t+0.5*args["h"], theta)
    k3 = args["h"]*f(args, r+0.5*k2, t+0.5*args["h"], theta)
    k4 = args["h"]*f(args, r+k3, t+args["h"], theta)
    return (k1 + 2*k2 + 2*k3 + k4)/6


# NORM 1 DISTANCE
def norm1_distance(u, v):
    return norm(u - v, 1, axis=1)


# Run simulation
def run_simulation(args, t_points, theta):
    aux_params = [args['E_0'], args['S_0']]

    r = expand_dims(array(aux_params, float), 0)

    r = repeat(r, args['N'], axis=0)

    for t in t_points:
        if t == 0.:
            P = r[:,[0]]
        else:
            P = concatenate((P, r[:,[0]]), 1)

        r += rk4(args, r, t, theta)

    return P


# LOADING DATA
def load_data(args, row_value=0):
    file_name = args['file']

    # LOAD DATA
    with open(file_name, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        counter = 0
        v = []
        for row in reader:
            if counter==row_value:
                for i in range(len(row)):
                    v.append(float(row[i]))
                break
            counter+= 1

    if len(v) == 0:
        raise Exception("Something wrong with data loading! Please check csv file.")

    # set as numpy array
    v = asarray(v)

    # how long the experiment lasts (max is 15 min)
    if args['exp_last'] > 15.:
        exp_last = 15.
    else:
        exp_last = args['exp_last']
    # get data
    print(exp_last)
    print(args['h'])
    print(int(ceil(exp_last/args['h'])))
    A = v[0:int(ceil(exp_last/args['h']))]
    # repeat data
    P_real = repeat( expand_dims(asarray(A),0), args['N'], axis=0 )
    # time points
    t_points = arange(0, args['h']*P_real.shape[1], args['h'])

    return P_real, t_points


# ABC PROCEDURE
def abc_procedure(args, P_real, t_point):
    accepted_theta = array([])

    distance = norm1_distance

    epsilons = [0.0001, 0.0002, 0.0003, 0.0005, 0.0007,
                0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009,
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
                0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    stop_sim = False

    # START
    print('Program runs...')

    for epsilon in epsilons:

        sys.stdout.write("o")
        sys.stdout.flush()

        for simulation_run in range(args['No_sim']):

            theta = zeros((args['N'], 2))
            theta[:,0] = choice(linspace(args['low_K_M'], args['high_K_M'], args['points_prior']), size=(theta.shape[0],))
            theta[:,1] = choice(linspace(args['low_k_cat'], args['high_k_cat'], args['points_prior']), size=(theta.shape[0],)) * 60.

            # x ~ simulator(theta)
            P_sim = run_simulation(args, t_point, theta)
            p_max = max(P_real)
            p_min = abs(min(P_real))
            d = distance((P_real-p_min)/(p_max-p_min), (P_sim-p_min)/(p_max-p_min))/P_sim.shape[1]

            if sum(d < epsilon) > 0:
                params = theta[d < epsilon, :]
                if accepted_theta.shape[0] == 0:
                    accepted_theta = params
                else:
                    accepted_theta = concatenate((accepted_theta, params), 0)

            if simulation_run == 25:
                if accepted_theta.shape[0] < 5:
                    accepted_theta = array([])
                    break
                else:
                    stop_sim = True

        if stop_sim:
            break

    if accepted_theta.shape[0] > 0:
        accepted_theta[:,1] /= 60.

    sys.stdout.write("\n")
    print('... and ends!')

    return accepted_theta


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
def config2args(config):
    args = {}
    args['file'] = str(config["file_name"]["file"])
    args['E_0'] = float(config["experiment_details"]["E_0"])
    args['S_0'] = float(config["experiment_details"]["S_0"])
    args['h'] = float(config["experiment_details"]["h"]) / 60.
    args['exp_last'] = float(config["experiment_details"]["exp_last"])

    args['low_K_M'] = float(config["abc_details"]["low_K_M"])
    args['high_K_M'] = float(config["abc_details"]["high_K_M"])
    args['low_k_cat'] = float(config["abc_details"]["low_k_cat"])
    args['high_k_cat'] = float(config["abc_details"]["high_k_cat"])

    args['N'] = int(config["abc_details"]["N"])
    args['No_sim'] = int(config["abc_details"]["No_sim"])
    args['points_prior'] = int(config["abc_details"]["points_prior"])

    return args

def main( ):

    file = str( input('Please provide path to the config file (e.g., my_config.json): ') )

    with open(file) as f:
        config = load(f)

    args = config2args(config)

    P_real, t = load_data(args)

    accepted_theta = abc_procedure(args, P_real, t)
    if accepted_theta.shape[0] == 0:
        raise Exception('There is no result found. Please check the following points:\n'
                        '1. Is the data properly provided?\n'
                        '2. Is the experiment set-up (E_0, S_0, h and exp_last) properly provided?\n'
                        '3. Please consider higher or lower values of high_K_M and high_k_cat. It might be that the real '
                        'value is outside one of these numbers. Please remember also that if you have any prior knowledge '
                        'about possible values of K_M or/and k_cat, better low and high values will drastically improve'
                        'the final result.')
    else:
        K_M_mean = mean(accepted_theta[:,[0]], 0, keepdims=False)
        K_M_std = std(accepted_theta[:,[0]], 0, keepdims=False)

        k_cat_mean = mean(accepted_theta[:,[1]], 0, keepdims=False)
        k_cat_std = std(accepted_theta[:,[1]], 0, keepdims=False)

        print('\n--Final results--\n'
              'k_cat\n'
              '> mean: {:.3f}\n'
              '> mean - std: {:.3f}\n'
              '> mean + std: {:.3f}\n\n'
              'K_M\n'
              '> mean: {:.3f}\n'
              '> mean - std: {:.3f}\n'
              '> mean + std: {:.3f}'.format(
            k_cat_mean[0],
            k_cat_mean[0] -  k_cat_std[0],
            k_cat_mean[0] +  k_cat_std[0],
            K_M_mean[0],
            K_M_mean[0] - K_M_std[0],
            K_M_mean[0] + K_M_std[0]))

        with open('results.txt', 'a') as f:
            date = strftime("%a, %d %b %Y %H:%M:%S", gmtime())
            print('Date of the experiment {}: {}\n'
            '--Final results--\n'
            'k_cat\n'
            '> mean: {:.3f}\n'
            '> mean - std: {:.3f}\n'
            '> mean + std: {:.3f}\n\n'
            'K_M\n'
            '> mean: {:.3f}\n'
            '> mean - std: {:.3f}\n'
            '> mean + std: {:.3f}\n'.format(
                file,
                date,
                k_cat_mean[0],
                k_cat_mean[0] - k_cat_std[0],
                k_cat_mean[0] + k_cat_std[0],
                K_M_mean[0],
                K_M_mean[0] - K_M_std[0],
                K_M_mean[0] + K_M_std[0]),
                file=f)

if __name__ == "__main__":
    main( )