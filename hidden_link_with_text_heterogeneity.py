"""

"""

import pandas as pd
import multiprocessing
import numpy as np

from numba import jit
import csv
import math
import datetime
import time
import scipy
import os



def is_constrained(E1, E2):
    return True


def constrained_clustering(C):
    return C


if __name__ == '__main__':
    # 1. given classified texts C
    # and two sets of observation samples at different times(T1,T2)
    C=[]

    while (True):
        # 2. calculate E1 and E2
        E1 = []
        E2 = []

        # 3. see if constrained
        constrained = is_constrained(E1, E2)
        if constrained:
            break

        # 4. update text classifications
        C = constrained_clustering(C)

    # 5. output results