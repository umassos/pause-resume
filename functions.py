import random
import math
from sympy import symbols, solve
import numpy as np
import pandas as pd

def randomInterval(df, T):
    randInd = random.randrange(len(df)-T)

    int = df[randInd:randInd+T]
    display(int)
    print(len(int))

    int['carbon_intensity_avg'].values.tolist()


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
        if prevAccepted != accept:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            runningList.append(vals[i])
            cost += vals[i]
        sol.append(runningList)

    return sol, cost


# list of costs (values)    -- vals
# length of job             -- k
# switching cost            -- beta
def kMinOnline(vals, k, U, L, beta):

    # solve for alpha
    a = symbols('a', real=True, positive=True)
    expr = (1 - L/U)/(1 - 1/a) - (1 + 1/(k*a))**k
    sol = solve(expr)
    if len(sol) < 1:
        print("something went wrong here")
    alpha = sol[0]

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
        if prevAccepted != accept:
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