# algorithm implementations for online search with switching penalty
# January 2023

import random
import math

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
def rodCuttingKmin(vals, k, beta):
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


if __name__ == '__main__':
    #main()
    vals = [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9]
    arr, cost = smallestSubsequenceK(vals, 4)
    print(arr)
    print(cost)

    sol, cost = rodCuttingKmin(vals, 8, 10)
    print(sol)
    print(cost)
