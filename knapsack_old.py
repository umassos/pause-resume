# knapsack of capacity W        -- W
# list of weights for each item -- weights
# list of values for each item  -- vals
# number of items               -- n
def dpOptimalKnapsack(W, weights, vals, n):
    dp = [0 for i in range(W + 1)]  # Making the dp array
    packed = set()

    for i in range(1, n + 1):  # taking first i elements
        for w in range(W, 0, -1):  # starting from back,so that we also have data of
            # previous computation when taking i-1 items
            if weights[i - 1] <= w:
                if dp[w - weights[i - 1]] + vals[i - 1] > dp[w]:
                    packed.add(i-1)
                dp[w] = max(dp[w], dp[w - weights[i - 1]] + vals[i - 1])

    return dp[W]  # returning the maximum value of knapsack, plus the packed values


        

# knapsack of capacity W                        -- W
# list of weights for each item                 -- weights
# list of values for each item                  -- vals
# number of items                               -- n
def randomOnlineKnapsack(W, weights, vals, n):
    a1packed = set()
    a2packed = set()
    a1value = 0
    a2value = 0
    a1remainingW, a2remainingW = (W, W)
    a1Halted = False
    a1utilization = []
    a2utilization = []
    a1profit = []
    a2profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        # algorithm 1 - basic greedy algorithm
        if not a1Halted:
            if (a1remainingW > weights[i]):
                a1packed.add(i)
                a1value += vals[i]
                a1remainingW -= weights[i]
            else:
                a1Halted = True
                a1utilization.append(W - a1remainingW)
                a1profit.append(a1value)
                a2utilization.append(W - a2remainingW)
                a2profit.append(a2value)
                continue
        # algorithm 2 - begins to greedily pack items after a1 halts
        elif a1Halted and (a2remainingW > weights[i]):
            a2packed.add(i)
            a2value += vals[i]
            a2remainingW -= weights[i]
        a1utilization.append(W - a1remainingW)
        a1profit.append(a1value)
        a2utilization.append(W - a1remainingW)
        a2profit.append(a2value)

    # randomly choose between algorithm 1 and 2, each with prob 0.5:
    if (random.random() < 0.5):
        return a1profit, a1utilization, a1packed
    else:
        return a2profit, a2utilization, a2packed


# knapsack of capacity W                        -- W
# list of weights for each item                 -- weights
# list of values for each item                  -- vals
# number of items                               -- n
# advice bit (true if there is weight > W/2)    -- advice
def adviceOnlineKnapsack(W, weights, vals, n, advice):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        # advice boolean becomes false as soon as we encounter the large item
        if advice:
            advice = (weights[i] < W/2)

        # greedy component -- when advice is false, we know that we've either
        # already encountered the big item, or there wasn't one to begin with.
        if not advice and (remainingW > weights[i]):
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the maximum value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
def boundedOnlineKnapsack(W, weights, vals, n, L, U):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        z_j = (W - remainingW) / W  # how much of knapsack is occupied
        phi_j = phi(z_j, L, U)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= phi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
def boundedRandomizedOnlineKnapsack(W, weights, vals, n, L, U):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        util = (W - remainingW) / W
        z_j = random.uniform(util, util*2) # instead of utilization, generate a random number
        phi_j = phi(z_j, L, U)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= phi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# knapsack of capacity W                -- W
# list of weights for each item         -- weights
# list of values for each item          -- vals
# number of items                       -- n
# lower bound on value size ratio       -- L
# upper bound on value size ratio       -- U
def boundedConstantRandomizedOnlineKnapsack(W, weights, vals, n, L, U):
    packed = set()
    value = 0
    remainingW = W
    utilization = []
    profit = []
    z_j = random.uniform(0, 1)  # instead of utilization, generate a random number

    #''simulate'' the behavior of online algorithm using a for loop
    for i in range(n):
        phi_j = phi(z_j, L, U)

        # add item if value/weight ratio is greater than phi
        if (vals[i]/weights[i]) >= phi_j and (remainingW - weights[i]) > 0 :
            packed.add(i)
            value += vals[i]
            remainingW -= weights[i]
        utilization.append(W - remainingW)
        profit.append(value)

    return profit, utilization, packed  # returning the value of knapsack, plus the packed values

# helper function phi for bounded online knapsack
def phi(z, L, U):
    return (((U*e)/L)**z)*(L/e)

def generatePLaw(n, xmax, alpha):
    s = []          # generate list of samples from power law distribution - lots of small values, few very large values
    for _ in range(n):
        s.append(int(xmax*((1-np.random.uniform(0,1))**(1/(1-alpha)))+1))
    return s

def generateGauss(n, mu, sigma):
    s = []          # generate list of samples from absolute value of gaussian distribution
    for _ in range(n):
        s.append(int(np.absolute(np.random.normal(mu, sigma))+1))
    return s

def drawPlot(optimal, algos, legend, title):
    for result in algos:
        plt.plot(range(len(result)), result, alpha = 0.5)
    plt.axhline(y=optimal, color='r', linestyle='-')
    plt.legend(legend)
    plt.title(title)
    plt.ylabel(title.split()[0])
    plt.xlabel("items presented to algorithm")
    plt.show()

def main():
    n = 10000

    # values and weights generated from power law distribution
    values = generatePLaw(n, 50, 0.8)
    dev = np.random.normal(0, 10, 1)[0]
    weights = generatePLaw(n, 50 + dev, 0.8)

    cap = int(sum(weights)/(n/8))
    advice = True if max(weights) >= cap/2 else False   # checks whether there is a large item in the set
    print(advice)

    optVal = dpOptimalKnapsack(cap, weights, values, n)

    adviceVals, adviceUtils, advicePacked = adviceOnlineKnapsack(cap, weights, values, n, advice)
    randVals, randUtils, randPacked = randomOnlineKnapsack(cap, weights, values, n)

    ratios = []
    for weight,val in zip(weights, values):
        ratios.append(val/weight)
    lower = min(ratios)
    upper = max(ratios)
    boundVals, boundUtils, boundPacked = boundedOnlineKnapsack(cap, weights, values, n, lower, upper)
    boundrandVals, boundrandUtils, boundrandPacked = boundedRandomizedOnlineKnapsack(cap, weights, values, n, lower, upper)

    legend = ["Advice Online Knapsack", "Randomized Online Knapsack", "Bounded Ratio Online Knapsack", "Bounded Randomized Online Knapsack", "Optimal Knapsack"]
    drawPlot(optVal, [adviceVals, randVals, boundVals, boundrandVals], legend, "Value in knapsack over time, different algorithms")
    drawPlot(cap, [adviceUtils, randUtils, boundUtils, boundrandUtils], legend, "Weight in knapsack over time, different algorithms")

    # values and weights of items generated from absolute value of gaussian
    values = generateGauss(n, 50, 10)
    dev = np.random.normal(0, 10, 1)[0]
    weights = generateGauss(n, 50 + dev, 10)

    cap = int(sum(weights)/(n/10))
    advice = True if max(weights) >= (cap / 2) else False    # checks whether there is a large item in the set
    print(advice)

    optVal = dpOptimalKnapsack(cap, weights, values, n)

    adviceVals, adviceUtils, advicePacked = adviceOnlineKnapsack(cap, weights, values, n, advice)
    randVals, randUtils, randPacked = randomOnlineKnapsack(cap, weights, values, n)

    ratios = []
    for weight, val in zip(weights, values):
        ratios.append(val / weight)
    lower = min(ratios)
    upper = max(ratios)
    boundVals, boundUtils, boundPacked = boundedOnlineKnapsack(cap, weights, values, n, lower, upper)
    boundrandVals, boundrandUtils, boundrandPacked = boundedConstantRandomizedOnlineKnapsack(cap, weights, values, n, lower, upper)

    drawPlot(optVal, [adviceVals, randVals, boundVals, boundrandVals], legend, "Value in knapsack over time, different algorithms")
    drawPlot(cap, [adviceUtils, randUtils, boundUtils, boundrandUtils], legend, "Weight in knapsack over time, different algorithms")