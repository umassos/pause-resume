import random
import math
from sympy import symbols, solve
import numpy as np
import pandas as pd
import pickle

def randomInterval(df, T, noiseFactor=0):
    randInd = random.randrange(len(df)-T)
    int = df[randInd:randInd+T]
    noise = np.random.normal(loc=0.0, scale=noiseFactor, size=T)
    interval = int['carbon_intensity_avg'].values
    interval = interval + noise
    interval = interval.clip(min=0)
    return interval.tolist()

def addNoise(vals, noiseFactor):
    noise = np.random.normal(loc=0.0, scale=noiseFactor, size=len(vals))
    noisyInterval = np.array(vals) + noise
    noisyInterval = noisyInterval.clip(min=0)
    return noisyInterval.tolist()

def boundedValues(df):
    L = df["carbon_intensity_avg"].min()
    U = df["carbon_intensity_avg"].max()
    return L, U

# list of values            -- vals
# length of subsequence     -- k
def smallestSubsequenceK(vals, k):
    subseq = []
    indices = (0,0)
    curSum = math.inf
    n = len(vals)

    for i in range(0, n-k+1):  
        subarray = vals[i:i+k]
        if sum(subarray) < curSum:
            subseq = subarray
            curSum = sum(subarray)
            indices = (i, i+k)

    return subseq, indices  # returning the min subsequence, and its indices


# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
def dynProgOPT(vals, k, beta):
    minCost = math.inf
    sol = []
    # n = len(vals)
    if (k == 0):
        return sol, 0

    for i in range(1, k+1):
        newVals = vals.copy()
        subseq, indices = smallestSubsequenceK(newVals, i) # get the smallest subsequence of length i

        # subtract subseq from vals
        del newVals[indices[0]:indices[1]]

        otherSeq, otherSum = dynProgOPT(newVals, k-i, beta)
        curCost = sum(subseq) + otherSum + 2*beta

        if curCost < minCost:
            minCost = curCost
            sol = []
            sol.append(subseq)
            for seq in otherSeq:
                sol.append(seq)
    
    return sol, minCost

# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
def carbonAgnostic(vals, k, beta):
    subseq = vals[0:k]
    cost = sum(subseq) + 2*beta
    
    return [subseq], cost

# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
def oneMinOnline(vals, k, U, L, beta):
    prevAccepted = False
    sol = []
    accepted = 0
    runningList = []
    lastElem = len(vals)
    cost = 0

    threshold = math.sqrt(U*L)

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if accepted + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break
        accept = (val <= threshold)
        if prevAccepted != accept:
            if len(runningList) > 1:
                sol.append(runningList)
            runningList = []
            cost += beta
        if accept:
            runningList.append(val)
            accepted += 1
            cost += val
            if accepted == k:
                sol.append(runningList)
                cost += beta # one last switching cost to turn off
                break
        prevAccepted = accept

    if accepted < k:
        if prevAccepted != True:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            runningList.append(vals[i])
            cost += vals[i]
        sol.append(runningList)

    return sol, cost

def computeAlphakmin(k, U, L):
    a = symbols('a', positive=True, real=True)
    expr = (1 - L/U)/(1 - 1/a) - (1 + 1/(k*a))**k
    sol = solve(expr)
    if len(sol) < 1:
        print("something went wrong here k={}".format(k))

    return sol[0]

def savedAlphakmin(k, U, L):
    memoKmin = pickle.load( open( "alphaKmin.pickle", "rb" ) )
    if (k, U, L) not in memoKmin.keys():
        print("something went wrong here {} {} {}".format(k, U, L))
    return memoKmin[(k,U,L)]


# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
def kMinOnline(vals, k, U, L, beta):

    # solve for alpha
    thresholds = []
    alpha = savedAlphakmin(k, U, L)
    thresholds = [ U*(1 - (1 - (1/alpha)) * (1 + (1/(alpha*k)))**(i-1) ) for i in range(1, k+1)]
    
    thres = iter(thresholds)

    prevAccepted = False
    sol = []
    accepted = 0
    runningList = []
    lastElem = len(vals)
    cost = 0

    threshold = next(thres)

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if accepted + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break
        accept = (val <= threshold)
        if prevAccepted != accept:
            if len(runningList) > 1:
                sol.append(runningList)
            runningList = []
            cost += beta
        if accept:
            runningList.append(val)
            accepted += 1
            cost += val
            if accepted == k:
                sol.append(runningList)
                cost += beta # one last switching cost to turn off
                break
            threshold = next(thres)
        prevAccepted = accept

    if accepted < k:
        if prevAccepted != True:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            runningList.append(vals[i])
            cost += vals[i]
        sol.append(runningList)

    return sol, cost

def computeAlphaPR(k, U, L, beta):    
    a = symbols('a', positive=True, real=True)
    expr = ((U-L-2*beta)/(U*(1 - 1/a) - 2*beta*(1-1/k-1/(k*a)))) - (1 + 1/(k*a))**k
    sol = solve(expr)
    if len(sol) < 1:
        print("something went wrong here k={}".format(k))

    return sol[0]

def savedAlphaPR(k, U, L, beta):
    memoPR = pickle.load( open( "alphaPR.pickle", "rb" ) )
    if (k, U, L, beta) not in memoPR.keys():
        print("something went wrong here {} {} {} {}".format(k, U, L, beta))
    return memoPR[(k, U, L, beta)]

# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
def pauseResumeOnline(vals, k, U, L, beta):
    # solve for alpha
    thresholds = []
    alpha = savedAlphaPR(k, U, L, beta)
    thresholds = [ U*(1 - (1 - (1/alpha)) * (1 + (1/(alpha*k)))**(i-1) ) + 2*beta*(((1/(k*alpha)) - (1/k) + 1) * (1 + (1/(alpha*k)))**(i-1)) for i in range(1, k+1)]
    
    thres = iter(thresholds)

    prevAccepted = False
    sol = []
    accepted = 0
    runningList = []
    lastElem = len(vals)
    cost = 0

    threshold = next(thres)

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if accepted + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break
        accept = False
        if prevAccepted == True and (val <= threshold):
            accept = True
        elif (val <= (threshold - 2*beta)):
            accept = True
        if prevAccepted != accept:
            if len(runningList) > 1:
                sol.append(runningList)
            runningList = []
            cost += beta
        if accept == True:
            runningList.append(val)
            accepted += 1
            cost += val
            if accepted == k:
                sol.append(runningList)
                cost += beta # one last switching cost to turn off
                break
            threshold = next(thres)
        prevAccepted = accept

    if accepted < k:
        if prevAccepted != True:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            runningList.append(vals[i])
            cost += vals[i]
        sol.append(runningList)

    return sol, cost

def generateSyntheticSequence(start, end, interval, noiseFactor=0):
    def cosine(x):
        return (5.1 + 5 * math.cos(x/3.8))
    
    sequence = []

    cur = start
    while (cur < end):
        sequence.append(cosine(cur))
        cur += interval

    noise = np.random.normal(0,noiseFactor,len(sequence))

    sequence = np.array(sequence) + noise

    sequence[sequence<0] = 0
    
    return sequence.tolist()