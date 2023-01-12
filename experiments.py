# algorithm implementations for online search with switching penalty
# January 2023

import random
import math
from sympy import symbols, solve
import numpy as np
import pandas as pd
import functions as f

if __name__ == '__main__':
    #main()
    k = 8
    U = 10
    L = 0.1
    switchCost = 2

    print("Testing on Synthetic Cosine Sequence")
    vals = f.generateSyntheticSequence(0, 48, 1)

    sol, cost = f.dynProgOPT(vals, k, beta=switch_cost)
    print(sol)
    print(cost)

    accepted, cost = f.oneMinOnline(vals, k, U, L, beta=switch_cost)
    print(accepted)
    print(cost)

    accepted, cost = f.kMinOnline(vals, k, U, L, beta=switch_cost)
    print(accepted)
    print(cost)


