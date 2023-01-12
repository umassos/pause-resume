# algorithm implementations for online search with switching penalty
# January 2023

import random
import math
from sympy import symbols, solve

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

        otherSeq, otherSum = rodCuttingKmin(newVals, k-i, beta)
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
    accepted = []
    lastElem = len(vals)
    cost = 0

    threshold = math.sqrt(U*L)

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if len(accepted) == k:
            break
        if len(accepted) + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break
        accept = (val <= threshold)
        if prevAccepted != accept:
            cost += beta
        if accept:
            accepted.append(val)
            cost += val
        prevAccepted = accept

    if len(accepted) < k:
        if prevAccepted != accept:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            accepted.append(vals[i])
            cost += vals[i]

    return accepted, cost


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
    accepted = []
    lastElem = len(vals)
    cost = 0

    threshold = next(thres)
    print(threshold)

    #simulate behavior of online algorithm using a for loop
    for (i, val) in enumerate(vals):
        if len(accepted) == k:
            break
        if len(accepted) + (len(vals)-i) == k: # must accept all remaining elements
            lastElem = i
            break
        accept = (val <= threshold)
        if prevAccepted != accept:
            cost += beta
        if accept:
            accepted.append(val)
            cost += val
            threshold = next(thres)
            print(threshold)
        prevAccepted = accept

    if len(accepted) < k:
        if prevAccepted != accept:
            cost += 2*beta
        for i in range(lastElem, len(vals)):
            accepted.append(vals[i])
            cost += vals[i]

    return accepted, cost



if __name__ == '__main__':
    #main()
    k = 8
    U = 10
    L = 1
    switchCost = 10

    vals = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
    arr, cost = smallestSubsequenceK(vals, 4)
    print(arr)
    print(cost)

    sol, cost = dynProgOPT(vals, k, beta=switchCost)
    print(sol)
    print(cost)

    accepted, cost = oneMinOnline(vals, k, U, L, beta=switchCost)
    print(accepted)
    print(cost)

    accepted, cost = kMinOnline(vals, k, U, L, beta=switchCost)
    print(accepted)
    print(cost)
