# Pause & Resume Experiments

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
3. **alpha*.pickle**: Caches a dictionary of precomputed alpha values for k-min search algorithm and online pause and resume algorithm
4. **carbon-traces/**: directory, contains carbon traces in .csv format.
    - "CA": "CA-ON.csv" (Ontario, Canada)
    - "NZ": "NZ-NZN.csv" (New Zealand)
    - "US": "US-NW-PACW.csv" (Pacific NW, USA)

# Reproducing Results

Given a correctly configured Python environment, with all dependencies, one can reproduce our results by cloning this repository, and running either of the following in a command line at the root directory, for synthetic and real-world networks, respectively:

``python experiments.py {TRACE CODE} {AMOUNT OF SLACK}``

Pass the abbreviation for the trace file and the desired amount of slack as command line arguments.  For example, running the experiments on the Ontario trace with 48 hours of slack is accomplished by running ``python experiments.py CA 48``
