#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math


def linear_alpha(start, bound, current_epoch, total_epoch):
    current_alpha = start - (start - bound) * current_epoch / total_epoch
    return current_alpha


def descend_alpha(start, bound, rate, current_epoch):
    current_alpha = start * math.pow(rate, current_epoch)
    if current_alpha > bound:
        return current_alpha
    else:
        return bound
