# algorithm implementations for online search with switching penalty
# January 2023

import sys
import functions as f
import random
import math
import itertools
from sympy import symbols, solve
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager, freeze_support
import seaborn as sns
import pickle

trace = sys.argv[1]
slack = sys.argv[2]

filename = ""
if trace == "NE":
    filename = "carbon-traces/US-CENT-SWPP.csv"
elif trace == "US":
    filename = "carbon-traces/US-NW-PACW.csv"
elif trace == "NZ":
    filename = "carbon-traces/NZ-NZN.csv"
elif trace == "CA":
    filename = "carbon-traces/CA-ON.csv"

# load a carbon trace
df = pd.read_csv(filename, parse_dates=["zone_datetime"])

############### EXPERIMENT SETUP ###############
trials = 25
# set U and L based on full year's worth of data
L, U = f.boundedValues(df)
# L = 11
# U = 1105

# set time horizon (e.g. slack) (in hours)
T = int(slack)

# fixed k, varying beta
k = 10
switches = range(0, int(U/5))

# fixed beta, varying k
switchCost = 50
ks = range(4, int(T/2))

noiseFactors = range(0, int(U/3), 5)

###############  ##############  ###############

vals = f.randomInterval(df, T)

def dynProgOPTMP(k, vals, switchCost, U, L):
    return f.dynProgOPT(vals, k, beta=switchCost)[1]
def agnosticMP(k, vals, switchCost, U, L):
    return f.carbonAgnostic(vals, k, beta=switchCost)[1]
def oneMinMP(k, vals, switchCost, U, L):
    return f.oneMinOnline(vals, k, U, L, beta=switchCost)[1]
def kMinMP(k, vals, switchCost, U, L):
    return f.kMinOnline(vals, k, U, L, beta=switchCost)[1]
def pauseResume(k, vals, switchCost, U, L):
    return f.pauseResumeOnline(vals, k, U, L, beta=switchCost)[1]

def dynProgOPTMP_unpack(args):
    return dynProgOPTMP(*args)
def agnosticMP_unpack(args):
    return agnosticMP(*args)
def oneMinMP_unpack(args):
    return oneMinMP(*args)
def kMinMP_unpack(args):
    return kMinMP(*args)
def pauseResume_unpack(args):
    return pauseResume(*args)

def computeAlphakmin_unpack(args):
    return f.computeAlphakmin(*args)
def computeAlphaPR_unpack(args):
    return f.computeAlphaPR(*args)

def prepAlphaExperiments(U, L, switchCost, switches, k, ks):
    memoKmin = pickle.load( open( "alphaKmin.pickle", "rb" ) )
    memoPR = pickle.load( open( "alphaPR.pickle", "rb" ) )

    # precompute alpha for each switching cost so we don't run into concurrency issues later
    print("precomputing alpha for all switching costs...")
    
    if (k, U, L) not in memoKmin.keys():
        memoKmin[(k, U, L)] = f.computeAlphakmin(k, U, L)
    pickle.dump( memoKmin, open( "alphaKmin.pickle", "wb" ) )

    to_be_computed = []
    for switch in switches:
        if (k, U, L, switch) not in memoPR.keys():
            to_be_computed.append(switch)

    with Pool(10) as p:
        alphas = p.map(computeAlphaPR_unpack, zip(itertools.repeat(k), itertools.repeat(U), itertools.repeat(L), to_be_computed))
        p.close()
        p.join()
    for (alpha, switch) in zip(alphas, to_be_computed):
        memoPR[(k, U, L, switch)] = alpha
    pickle.dump( memoPR, open( "alphaPR.pickle", "wb" ) )

    ## precompute alpha for each k so we don't run into concurrency issues later
    print("precomputing alpha for all k...")
    to_be_computed = []
    for k in ks:
        if (k, U, L) not in memoKmin.keys():
            to_be_computed.append(k)
    
    with Pool(10) as p:
        alphas = p.map(computeAlphakmin_unpack, zip(to_be_computed, itertools.repeat(U), itertools.repeat(L)))
        p.close()
        p.join()
    for (alpha, k) in zip(alphas, to_be_computed):
        memoKmin[(k, U, L)] = alpha
    pickle.dump( memoKmin, open( "alphaKmin.pickle", "wb" ) )

    to_be_computed = []
    for k in ks:
        if (k, U, L, switchCost) not in memoPR.keys():
            to_be_computed.append(k)

    with Pool(10) as p:
        alphas = p.map(computeAlphaPR_unpack, zip(to_be_computed, itertools.repeat(U), itertools.repeat(L), itertools.repeat(switchCost)))
        p.close()
        p.join()
    for (alpha, k) in zip(alphas, to_be_computed):
        memoPR[(k, U, L, switchCost)] = alpha
    pickle.dump( memoPR, open( "alphaPR.pickle", "wb" ) )

opts = []
agnosts = []
onemins = []
kmins = []
pauseres = []
if __name__ == '__main__':
    print("Testing diff. switch costs on " + trace + " carbon trace. Slack {} hrs & Job Length {} hrs".format(T, k))

    competitiveAgnostic = []
    competitive1min = []
    competitivekmin = []
    competitivePauseResume = []

    prepAlphaExperiments(U, L, switchCost, switches, k, ks)

    for _ in range(trials):
        sys.stdout.write("Trial {} -- OPT ".format(_))
        sys.stdout.flush()
        with Pool(10) as p:
            o = p.map(dynProgOPTMP_unpack, zip(itertools.repeat(k), itertools.repeat(vals), switches, itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- Agnostic ")
        sys.stdout.flush()
        with Pool(10) as p:
            a = p.map(agnosticMP_unpack, zip(itertools.repeat(k), itertools.repeat(vals), switches, itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- 1-min ")
        sys.stdout.flush()
        with Pool(10) as p:
            om = p.map(oneMinMP_unpack, zip(itertools.repeat(k), itertools.repeat(vals), switches, itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- k-min ")
        sys.stdout.flush()
        with Pool(10) as p:
            km = p.map(kMinMP_unpack, zip(itertools.repeat(k), itertools.repeat(vals), switches, itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- pause/resume ")
        sys.stdout.flush()
        with Pool(10) as p:
            pr = p.map(pauseResume_unpack, zip(itertools.repeat(k), itertools.repeat(vals), switches, itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        opts.append(o)
        agnosts.append(a)
        onemins.append(om)
        kmins.append(km)
        pauseres.append(pr)
        print("")
        vals = f.randomInterval(df, T)

    print("")

    # calculate competitive ratios
    opts = np.array(opts)
    agnosts = np.array(agnosts)
    onemins = np.array(onemins)
    kmins = np.array(kmins)
    pauseres = np.array(pauseres)
    crAgnostic = np.mean(agnosts/opts, axis = 0)
    crOnemin = np.mean(onemins/opts, axis = 0)
    crKmin = np.mean(kmins/opts, axis = 0)
    crPauseR = np.mean(pauseres/opts, axis = 0)
    competitiveAgnostic.append(crAgnostic)
    competitive1min.append(crOnemin)
    competitivekmin.append(crKmin)
    competitivePauseResume.append(crPauseR)

    legend = ["Carbon-Agnostic", "1-min Search", "k-min Search", "Pause & Resume (minimization)"]
    plt.plot(switches, crAgnostic, marker=".")
    plt.plot(switches, crOnemin, marker=".")
    plt.plot(switches, crKmin, marker=".")
    plt.plot(switches, crPauseR, marker=".")
    plt.legend(legend)
    plt.ylabel('Competitive Ratio')
    plt.xlabel("switching cost")
    plt.title("Testing switching costs on " + trace + " carbon trace. Slack {} hrs & Job Length {} hrs".format(T, k))
    plt.savefig("crBeta.pdf", facecolor='w', transparent=False)
    plt.clf()

    print("competitive ratios: ")
    print("carbon-agnostic: {}".format(np.mean(crAgnostic)))
    print("1-min search: {}".format(np.mean(crOnemin)))
    print("k-min search: {}".format(np.mean(crKmin)))
    print("pause & resume minimization: {}".format(np.mean(crPauseR)))

    opts = []
    agnosts = []
    onemins = []
    kmins = []
    pauseres = []
    print("Testing diff. job lengths on " + trace + " carbon trace. Slack {} hrs".format(T))
    
    for _ in range(trials):
        sys.stdout.write("Trial {} -- OPT ".format(_))
        sys.stdout.flush()
        with Pool(10) as p:
            o = p.map(dynProgOPTMP_unpack, zip(ks, itertools.repeat(vals), itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- Agnostic ")
        sys.stdout.flush()
        with Pool(10) as p:
            a = p.map(agnosticMP_unpack, zip(ks, itertools.repeat(vals), itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- 1-min ")
        sys.stdout.flush()
        with Pool(10) as p:
            om = p.map(oneMinMP_unpack, zip(ks, itertools.repeat(vals), itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- k-min ")
        sys.stdout.flush()
        with Pool(10) as p:
            km = p.map(kMinMP_unpack, zip(ks, itertools.repeat(vals), itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- pause/resume ")
        sys.stdout.flush()
        with Pool(10) as p:
            pr = p.map(pauseResume_unpack, zip(ks, itertools.repeat(vals), itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        opts.append(o)
        agnosts.append(a)
        onemins.append(om)
        kmins.append(km)
        pauseres.append(pr)
        print("")

        vals = f.randomInterval(df, T)

    print("")
    # calculate competitive ratios
    # calculate competitive ratios
    opts = np.array(opts)
    agnosts = np.array(agnosts)
    onemins = np.array(onemins)
    kmins = np.array(kmins)
    pauseres = np.array(pauseres)
    crAgnostic = np.mean(agnosts/opts, axis = 0)
    crOnemin = np.mean(onemins/opts, axis = 0)
    crKmin = np.mean(kmins/opts, axis = 0)
    crPauseR = np.mean(pauseres/opts, axis = 0)
    competitiveAgnostic.append(crAgnostic)
    competitive1min.append(crOnemin)
    competitivekmin.append(crKmin)
    competitivePauseResume.append(crPauseR)

    legend = ["Carbon-Agnostic", "1-min Search", "k-min Search", "Pause & Resume (minimization)"]
    length = len(ks)
    plt.plot(ks, crAgnostic, marker=".")
    plt.plot(ks, crOnemin, marker=".")
    plt.plot(ks, crKmin, marker=".")
    plt.plot(ks, crPauseR, marker=".")
    plt.legend(legend)
    plt.ylabel('Competitive Ratio')
    plt.xlabel("job length")
    plt.title("Testing diff. job lengths on " + trace + " carbon trace. Slack {} hrs & Switch Cost {}".format(T, switchCost))
    plt.savefig("crK.pdf", facecolor='w', transparent=False)
    plt.clf()

    print("competitive ratios: ")
    print("carbon-agnostic: {}".format(np.mean(crAgnostic)))
    print("1-min search: {}".format(np.mean(crOnemin)))
    print("k-min search: {}".format(np.mean(crKmin)))
    print("pause & resume minimization: {}".format(np.mean(crPauseR)))

    opts = []
    agnosts = []
    onemins = []
    kmins = []
    pauseres = []
    print("Testing diff. noise factors on " + trace + " carbon trace. Slack {} hrs".format(T))

    vals = f.randomInterval(df, T)
    valArray = [f.addNoise(vals, x) for x in noiseFactors]
    
    for _ in range(trials):
        sys.stdout.write("Trial {} -- OPT ".format(_))
        sys.stdout.flush()
        with Pool(10) as p:
            o = p.map(dynProgOPTMP_unpack, zip(itertools.repeat(k), valArray, itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- Agnostic ")
        sys.stdout.flush()
        with Pool(10) as p:
            a = p.map(agnosticMP_unpack, zip(itertools.repeat(k), valArray, itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- 1-min ")
        sys.stdout.flush()
        with Pool(10) as p:
            om = p.map(oneMinMP_unpack, zip(itertools.repeat(k), valArray, itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- k-min ")
        sys.stdout.flush()
        with Pool(10) as p:
            km = p.map(kMinMP_unpack, zip(itertools.repeat(k), valArray, itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        sys.stdout.write("-- pause/resume ")
        sys.stdout.flush()
        with Pool(10) as p:
            pr = p.map(pauseResume_unpack, zip(itertools.repeat(k), valArray, itertools.repeat(switchCost), itertools.repeat(U), itertools.repeat(L)))
            p.close()
            p.join()
        opts.append(o)
        agnosts.append(a)
        onemins.append(om)
        kmins.append(km)
        pauseres.append(pr)
        print("")

        vals = f.randomInterval(df, T)
        valArray = [f.addNoise(vals, x) for x in noiseFactors]

    print("")
    # calculate competitive ratios
    opts = np.array(opts)
    agnosts = np.array(agnosts)
    onemins = np.array(onemins)
    kmins = np.array(kmins)
    pauseres = np.array(pauseres)
    crAgnostic = np.mean(agnosts/opts, axis = 0)
    crOnemin = np.mean(onemins/opts, axis = 0)
    crKmin = np.mean(kmins/opts, axis = 0)
    crPauseR = np.mean(pauseres/opts, axis = 0)
    competitiveAgnostic.append(crAgnostic)
    competitive1min.append(crOnemin)
    competitivekmin.append(crKmin)
    competitivePauseResume.append(crPauseR)

    length = len(noiseFactors)
    plt.plot(noiseFactors, crAgnostic, marker=".")
    plt.plot(noiseFactors, crOnemin, marker=".")
    plt.plot(noiseFactors, crKmin, marker=".")
    plt.plot(noiseFactors, crPauseR, marker=".")
    plt.legend(legend)
    plt.ylabel('Competitive Ratio')
    plt.xlabel("noise factor")
    plt.title("Testing diff. noise factors on " + trace + " carbon trace. Slack {} hrs & Job Length {} hrs".format(T, k))
    plt.savefig("crNoise.pdf", facecolor='w', transparent=False)
    plt.clf()

    # CDF plot for competitive ratio (across all experiments)
    for twoDList in [competitiveAgnostic, competitive1min, competitivekmin, competitivePauseResume]:
        print(twoDList)
        data = []
        for list in twoDList:
            for item in list:
                data.append(item)
        # # No of data points used
        # N = len(data)
                
        # # sort the data in ascending order
        # x = np.sort(data)
        
        # # get the cdf values of y
        # y = np.arange(N) / float(N)
        # plt.plot(x, y)
        sns.kdeplot(data = data, cumulative = True)

    plt.legend(legend)
    plt.ylabel('cumulative probability')
    plt.xlabel("Competitive Ratio")
    #plt.xlim(0, 10)
    plt.title("CDF of Competitive Ratio, " + trace + " trace. Slack {} hrs & Switch Cost {}".format(T, switchCost))
    plt.savefig("cdf.pdf", facecolor='w', transparent=False)
    plt.clf()

# placeholder for 6 figures

# changing switching cost

# changing k
    # changing slack (ratio between k and T)

# changing noise

# changing U/L (changing location)



