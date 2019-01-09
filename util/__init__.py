#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    return
