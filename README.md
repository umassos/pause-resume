# The Online Pause and Resume Problem: Optimal Algorithms and An Application to Carbon-Aware Load Shifting

[<img src="https://img.shields.io/badge/Full%20Paper-2303.17551-B31B1B.svg?style=flat-square&logo=arxiv" height="25">](https://arxiv.org/abs/2303.17551)

We introduce and study the online pause and resume problem. In this problem, a player attempts to find the $k$ lowest (alternatively, highest) prices in a sequence of fixed length $T$, which is revealed sequentially. At each time step, the player is presented with a price and decides whether to accept or reject it. The player incurs a switching cost whenever their decision changes in consecutive time steps, i.e., whenever they pause or resume purchasing. This online problem is motivated by the goal of carbon-aware load shifting, where a workload may be paused during periods of high carbon intensity and resumed during periods of low carbon intensity and incurs a cost when saving or restoring its state. It has strong connections to existing problems studied in the literature on online optimization, though it introduces unique technical challenges that prevent the direct application of existing algorithms. Extending prior work on threshold-based algorithms, we introduce double-threshold algorithms for both the minimization and maximization variants of this problem. We further show that the competitive ratios achieved by these algorithms are the best achievable by any deterministic online algorithm. Finally, we empirically validate our proposed algorithm through case studies on the application of carbon-aware load shifting using real carbon trace data and existing baseline algorithms.

# Python code 

Our experimental code has been written in Python3.  We recommend using a tool to manage Python virtual environments, such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html).  There are several required Python packages:
- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [SymPy](https://www.sympy.org/en/index.html)
- [Matplotlib](https://matplotlib.org) for creating plots 
- [Seaborn](https://seaborn.pydata.org)

# Files and Descriptions

1. **functions.py**: Implements helper functions and algorithms.
2. **experiments.py**: Main Python script for all experiments.
3. **alpha\*.pickle**: Caches a dictionary of precomputed alpha values for k-min search algorithm and online pause and resume algorithm
4. **carbon-traces/**: directory, contains carbon traces in .csv format.
    - "CA": "CA-ON.csv" (Ontario, Canada)
    - "NZ": "NZ-NZN.csv" (New Zealand)
    - "US": "US-NW-PACW.csv" (Pacific NW, USA)

# Reproducing Results

Given a correctly configured Python environment, with all dependencies, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for synthetic and real-world networks, respectively:

``python experiments.py {TRACE CODE} {AMOUNT OF SLACK}``

Pass the abbreviation for the trace file and the desired amount of slack as command line arguments.  For example, running the experiments on the Ontario trace with 48 hours of slack is accomplished by running ``python experiments.py CA 48``

# Citation

> @article{lechowicz2023pauseresume,
> title={The Online Pause and Resume Problem: Optimal Algorithms and An Application to Carbon-Aware Load Shifting},
> volume={7},
> ISSN={2476-1249},
> url={http://dx.doi.org/10.1145/3626776},
> DOI={10.1145/3626776},
> number={3},
> journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems},
> publisher={Association for Computing Machinery (ACM)},
> author={Lechowicz, Adam and Christianson, Nicolas and Zuo, Jinhang and Bashir, Noman and Hajiesmaili, Mohammad and Wierman, Adam and Shenoy, Prashant},
> year={2023}, month={Dec}, pages={1–32} }
