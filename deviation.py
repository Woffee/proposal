#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os

def get_deviation(obs_original_file, obs_estimate_file):
    """
    计算原始 obs 和估计的 obs 之间的deviation。
    """
    obs_o = np.loadtxt(obs_original_file, delimiter=',')
    obs_e = np.loadtxt(obs_estimate_file, delimiter=',')

    # print(obs_e.shape)
    ss = 0
    for i in range(obs_o.shape[1]):
        oo = obs_o[:, i]
        ee = obs_e[:, i]
        ss = ss + sum( (oo - ee) ** 2)
    return 1.0 * ss / (obs_o.shape[0] * obs_o.shape[1])



