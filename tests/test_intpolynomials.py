"""
    Beta Expansions of Salem Numbers, calculating periods thereof
    Copyright (C) 2021 Michael P. Lane

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
"""

import copy
from itertools import product
from pathlib import Path
from unittest import TestCase

import numpy as np
from mpmath import mp, mpf, almosteq, workdps

from intpolynomials.intpolynomials import Int_Polynomial, Int_Polynomial_Array

def eval_code_in_file(filename, dps = 32):
    filename = Path(filename)
    with workdps(dps):
        with filename.open("r") as fh:
            return eval("".join(fh.readlines()))

def inequal_dps(x,y,max_dps = 256):
    for dps in range(1,max_dps+1):
        with workdps(dps):
            if not almosteq(x,y):
                return dps
    return 0

class Test_Int_Polynomial(TestCase):

    def setUp(self):

        mp.dps = 16

        pi =    mpf('3.141592653589793227')
        pi2 =   mpf('6.283185307179586454')
        pipi =  mpf('9.869604401089358436')
        pipi2 = mpf('19.73920880217871687')
        pi4 =   mpf('12.56637061435917291')

        smaller = mpf("0.000000001")
        small = mpf("0.001")

        good_polys = [
            (([0],                16, True), -1, ([0],       16, True),    [0],         [0],
             [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
             [(-pi, 0, 0), (pi, 0, 0)]),
            (([0, 0],             16, True), -1, ([0],       16, True),    [0],         [0],
             [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
             [(-pi, 0, 0), (pi, 0, 0)]),
            (([0, 0, 0],          16, True), -1, ([0],       16, True),    [0],         [0],
             [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
             [(-pi, 0, 0), (pi, 0, 0)]),

            (([0],                16, False),-1, ([0],       16, False),   [0],         [0],
             [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
             [(-pi, 0, 0), (pi, 0, 0)]),
            (([0, 0],             16, False),-1, ([0],       16, False),   [0],         [0],
             [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
             [(-pi, 0, 0), (pi, 0, 0)]),
            (([0, 0, 0],          16, False),-1, ([0],       16, False),   [0],         [0],
             [(0, 0, 0), (-1, 0, 0), (1, 0, 0)],
             [(-pi, 0, 0), (pi, 0, 0)]),

            (([1],                16, True),  0, ([1],       16, True),    [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),
            (([1, 0],             16, True),  0, ([1],       16, True),    [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),
            (([1, 0, 0],          16, True),  0, ([1],       16, True),    [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),
            (([1, 0, 0, 0],       16, True),  0, ([1],       16, True),    [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),

            (([1],                16, False), 0, ([1],       16, False),   [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),
            (([0, 1],             16, False), 0, ([1],       16, False),   [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),
            (([0, 0, 1],          16, False), 0, ([1],       16, False),   [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),
            (([0, 0, 0, 1],       16, False), 0, ([1],       16, False),   [1],         [1],
             [(0, 1, 0), (-1, 1, 0), (1, 1, 0)],
             [(-pi, 1, 0), (pi, 1, 0)]),

            (([2],                16, True),  0, ([2],       16, True),    [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),
            (([2, 0],             16, True),  0, ([2],       16, True),    [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),
            (([2, 0, 0],          16, True),  0, ([2],       16, True),    [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),
            (([2, 0, 0, 0],       16, True),  0, ([2],       16, True),    [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),

            (([2],                16, False), 0, ([2],       16, False),   [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),
            (([0, 2],             16, False), 0, ([2],       16, False),   [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),
            (([0, 0, 2],          16, False), 0, ([2],       16, False),   [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),
            (([0, 0, 0, 2],       16, False), 0, ([2],       16, False),   [2],         [2],
             [(0, 2, 0), (-1, 2, 0), (1, 2, 0)],
             [(-pi, 2, 0), (pi, 2, 0)]),

            (([-1],                16, True),  0, ([-1],       16, True),    [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),
            (([-1, 0],             16, True),  0, ([-1],       16, True),    [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),
            (([-1, 0, 0],          16, True),  0, ([-1],       16, True),    [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),
            (([-1, 0, 0, 0],       16, True),  0, ([-1],       16, True),    [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),

            (([-1],                16, False), 0, ([-1],       16, False),   [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),
            (([0, -1],             16, False), 0, ([-1],       16, False),   [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),
            (([0, 0, -1],          16, False), 0, ([-1],       16, False),   [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),
            (([0, 0, 0, -1],       16, False), 0, ([-1],       16, False),   [-1],         [-1],
             [(0, -1, 0), (-1, -1, 0), (1, -1, 0)],
             [(-pi, -1, 0), (pi, -1, 0)]),

            (([0, 1],             16, True),  1, ([0, 1],    16, True),    [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),
            (([0, 1, 0],          16, True),  1, ([0, 1],    16, True),    [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),
            (([0, 1, 0, 0],       16, True),  1, ([0, 1],    16, True),    [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),
            (([0, 1, 0, 0, 0],    16, True),  1, ([0, 1],    16, True),    [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),

            (([1, 0],             16, False), 1, ([1, 0],    16, False),   [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),
            (([0, 1, 0],          16, False), 1, ([1, 0],    16, False),   [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),
            (([0, 0, 1, 0],       16, False), 1, ([1, 0],    16, False),   [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),
            (([0, 0, 0, 1, 0],    16, False), 1, ([1, 0],    16, False),   [0, 1],      [1, 0],
             [(0, 0, 1), (-2, -2, 1), (-1, -1, 1), (1, 1, 1), (2, 2, 1)],
             [(-pi, -pi, 1), (pi, pi, 1)]),

            (([0, 2],             16, True),  1, ([0, 2],    16, True),    [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),
            (([0, 2, 0],          16, True),  1, ([0, 2],    16, True),    [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),
            (([0, 2, 0, 0],       16, True),  1, ([0, 2],    16, True),    [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),
            (([0, 2, 0, 0, 0],    16, True),  1, ([0, 2],    16, True),    [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),

            (([2, 0],             16, False), 1, ([2, 0],    16, False),   [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),
            (([0, 2, 0],          16, False), 1, ([2, 0],    16, False),   [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),
            (([0, 0, 2, 0],       16, False), 1, ([2, 0],    16, False),   [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),
            (([0, 0, 0, 2, 0],    16, False), 1, ([2, 0],    16, False),   [0, 2],      [2, 0],
             [(0, 0, 2), (-2, -4, 2), (-1, -2, 2), (1, 2, 2), (2, 4, 2)],
             [(-pi, -pi2, 2), (pi, pi2, 2)]),

            (([0, -1],             16, True),  1, ([0, -1],    16, True),    [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),
            (([0, -1, 0],          16, True),  1, ([0, -1],    16, True),    [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),
            (([0, -1, 0, 0],       16, True),  1, ([0, -1],    16, True),    [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),
            (([0, -1, 0, 0, 0],    16, True),  1, ([0, -1],    16, True),    [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),

            (([-1, 0],             16, False), 1, ([-1, 0],    16, False),   [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),
            (([0, -1, 0],          16, False), 1, ([-1, 0],    16, False),   [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),
            (([0, 0, -1, 0],       16, False), 1, ([-1, 0],    16, False),   [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),
            (([0, 0, 0, -1, 0],    16, False), 1, ([-1, 0],    16, False),   [0, -1],      [-1, 0],
             [(0, 0, -1), (-2, 2, -1), (-1, 1, -1), (1, -1, -1), (2, -2, -1)],
             [(-pi, pi, -1), (pi, -pi, -1)]),

            (([0, 0, 1],          16, True),  2, ([0, 0, 1], 16, True),    [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),
            (([0, 0, 1, 0],       16, True),  2, ([0, 0, 1], 16, True),    [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),
            (([0, 0, 1, 0, 0],    16, True),  2, ([0, 0, 1], 16, True),    [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),
            (([0, 0, 1, 0, 0, 0], 16, True),  2, ([0, 0, 1], 16, True),    [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),

            (([1, 0, 0],          16, False), 2, ([1, 0, 0], 16, False),   [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),
            (([0, 1, 0, 0],       16, False), 2, ([1, 0, 0], 16, False),   [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),
            (([0, 0, 1, 0, 0],    16, False), 2, ([1, 0, 0], 16, False),   [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),
            (([0, 0, 0, 1, 0, 0], 16, False), 2, ([1, 0, 0], 16, False),   [0, 0, 1],   [1, 0, 0],
             [(0, 0, 0), (-3, 9, -6), (-2, 4, -4), (-1, 1, -2), (1, 1, 2), (2, 4, 4), (3, 9, 6)],
             [(-pi, pipi, -pi2), (pi, pipi, pi2)]),

            (([0, 0, 2],          16, True),  2, ([0, 0, 2], 16, True),    [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),
            (([0, 0, 2, 0],       16, True),  2, ([0, 0, 2], 16, True),    [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),
            (([0, 0, 2, 0, 0],    16, True),  2, ([0, 0, 2], 16, True),    [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),
            (([0, 0, 2, 0, 0, 0], 16, True),  2, ([0, 0, 2], 16, True),    [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),

            (([2, 0, 0],          16, False), 2, ([2, 0, 0], 16, False),   [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),
            (([0, 2, 0, 0],       16, False), 2, ([2, 0, 0], 16, False),   [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),
            (([0, 0, 2, 0, 0],    16, False), 2, ([2, 0, 0], 16, False),   [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),
            (([0, 0, 0, 2, 0, 0], 16, False), 2, ([2, 0, 0], 16, False),   [0, 0, 2],   [2, 0, 0],
             [(0, 0, 0), (-3, 18, -12), (-2, 8, -8), (-1, 2, -4), (1, 2, 4), (2, 8, 8), (3, 18, 12)],
             [(-pi, pipi2, -pi4), (pi, pipi2, pi4)]),

            (([0, 0, -1],          16, True),  2, ([0, 0, -1], 16, True),    [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3,-9, -6)],
             [(-pi,-pipi,pi2), (pi, -pipi, -pi2)]),
            (([0, 0, -1, 0],       16, True),  2, ([0, 0, -1], 16, True),    [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),
            (([0, 0, -1, 0, 0],    16, True),  2, ([0, 0, -1], 16, True),    [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),
            (([0, 0, -1, 0, 0, 0], 16, True),  2, ([0, 0, -1], 16, True),    [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),

            (([-1, 0, 0],          16, False), 2, ([-1, 0, 0], 16, False),   [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),
            (([0, -1, 0, 0],       16, False), 2, ([-1, 0, 0], 16, False),   [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),
            (([0, 0, -1, 0, 0],    16, False), 2, ([-1, 0, 0], 16, False),   [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),
            (([0, 0, 0, -1, 0, 0], 16, False), 2, ([-1, 0, 0], 16, False),   [0, 0, -1],   [-1, 0, 0],
             [(0, 0, 0), (-3, -9, 6), (-2, -4, 4), (-1, -1, 2), (1, -1, -2), (2, -4, -4), (3, -9, -6)],
             [(-pi, -pipi, pi2), (pi, -pipi, -pi2)]),

            (([1, 0, 0, 0, 0, 1],  16, True),  5, ([1, 0, 0, 0, 0, 1], 16, True), [1, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 1],
             [(0, 1, 0), (-3,-242, 405), (-2, -31, 80), (-1, 0, 5), (1, 2, 5), (2, 33, 80), (3, 244, 405)],
             [(smaller, 1, 0), (-smaller, 1, 0),
              (small, mpf('1.000000000000000999'), mpf('4.999999999999999899e-12')), (-small, mpf('0.9999999999999990008'), mpf('4.999999999999999899e-12')),
              (pi, mpf('307.0196847852814486'), mpf('487.0454551700121826')), (-pi, mpf('-305.0196847852814486'), mpf('487.0454551700121826'))
             ])
        ]

        self.good_polys = {
            "poly1_args":       list(map(lambda t: t[0], good_polys)),
            "deg"  :            list(map(lambda t: t[1], good_polys)),
            "poly2_args":       list(map(lambda t: t[2], good_polys)),
            "list_natural":     list(map(lambda t: t[3], good_polys)),
            "list_unnatural":   list(map(lambda t: t[4], good_polys)),
            "int_inputs":       list(map(lambda t: list(map(lambda s: (s[0], mpf(s[1]), mpf(s[2])), t[5])), good_polys)),
            "mpf_inputs":       list(map(lambda t: list(map(lambda s: (s[0], mpf(s[1]), mpf(s[2])), t[6])), good_polys)),
        }

        self.good_polys["array_natural"] = list(map(lambda lst: np.array(lst, dtype = np.longlong), self.good_polys["list_natural"]))
        self.good_polys["array_unnatural"] = list(map(lambda lst: np.array(lst, dtype = np.longlong), self.good_polys["list_unnatural"]))

        self.bad_polys = [
            ([], 16, True),
            ([], 16, False)
        ]

    def test___init__(self):
        for args, deg in zip(self.good_polys["poly1_args"], self.good_polys["deg"]):
            try:
                Int_Polynomial(*args)
            except Exception as m:
                self.fail("Did not construct successfully; exception info: %s" % m)
            if deg >= 0:
                self.assertNotEqual(Int_Polynomial(*args)[deg], 0, str(args))
        for args in self.bad_polys:
            with self.assertRaises(ValueError):
                Int_Polynomial(*args)

    def test_get_deg(self):
        for args, deg in zip(self.good_polys["poly1_args"], self.good_polys["deg"]):
            self.assertEqual(
                Int_Polynomial(*args).get_deg(),
                deg,
                "args: %s" % (args,)
            )

    def test_trim(self):
        for args, deg in zip(self.good_polys["poly1_args"], self.good_polys["deg"]):pass

    def test_array_coefs(self):
        for args, lst in zip(self.good_polys["poly1_args"], self.good_polys["array_natural"]):
            self.assertTrue(np.all(Int_Polynomial(*args).ndarray_coefs() == lst))
            self.assertTrue(Int_Polynomial(*args).ndarray_coefs().dtype == np.longlong)
        for args, lst in zip(self.good_polys["poly1_args"], self.good_polys["array_unnatural"]):
            self.assertTrue(np.all(Int_Polynomial(*args).ndarray_coefs(False) == lst))

    def test___eq__(self):
        for args1, args2 in zip(self.good_polys["poly1_args"], self.good_polys["poly2_args"]):
            with self.subTest():
                self.assertEqual(Int_Polynomial(*args1), Int_Polynomial(*args2), str((args1, args2)))
        for args in self.good_polys["poly1_args"]:

            diff_args = copy.deepcopy(args)
            coefs = diff_args[0]
            coefs[0] += 1
            with self.subTest():
                self.assertNotEqual(Int_Polynomial(*args), Int_Polynomial(*diff_args), str(args))

            if len(args[0]) >= 2:
                diff_args = copy.deepcopy(args)
                coefs = diff_args[0]
                coefs[1] += 1
                with self.subTest():
                    self.assertNotEqual(Int_Polynomial(*args), Int_Polynomial(*diff_args), str(args))

            if len(args[0]) >= 3:
                diff_args = copy.deepcopy(args)
                coefs = diff_args[0]
                coefs[2] += 1
                with self.subTest():
                    self.assertNotEqual(Int_Polynomial(*args), Int_Polynomial(*diff_args), str(args))

            diff_args = copy.deepcopy(args)
            diff_args = (list(reversed(diff_args[0])), diff_args[1], not diff_args[2])
            with self.subTest():
                self.assertEqual(Int_Polynomial(*args), Int_Polynomial(*diff_args), str(args))

    def test___hash__(self):
        for args1, args2 in zip(self.good_polys["poly1_args"], self.good_polys["poly2_args"]):
            with self.subTest():
                self.assertEqual(hash(Int_Polynomial(*args1)), hash(Int_Polynomial(*args2)), str((args1, args2)))
        for args in self.good_polys["poly1_args"]:

            diff_args = copy.deepcopy(args)
            coefs = diff_args[0]
            coefs[0] += 1
            with self.subTest():
                self.assertNotEqual(hash(Int_Polynomial(*args)), hash(Int_Polynomial(*diff_args)), str(args))

            if len(args[0]) >= 2:
                diff_args = copy.deepcopy(args)
                coefs = diff_args[0]
                coefs[1] += 1
                with self.subTest():
                    self.assertNotEqual(hash(Int_Polynomial(*args)), hash(Int_Polynomial(*diff_args)), str(args))

            if len(args[0]) >= 3:
                diff_args = copy.deepcopy(args)
                coefs = diff_args[0]
                coefs[2] += 1
                with self.subTest():
                    self.assertNotEqual(hash(Int_Polynomial(*args)), hash(Int_Polynomial(*diff_args)), str(args))

            diff_args = copy.deepcopy(args)
            diff_args = (list(reversed(diff_args[0])), diff_args[1], not diff_args[2])
            with self.subTest():
                self.assertEqual(hash(Int_Polynomial(*args)), hash(Int_Polynomial(*diff_args)), str(args))

    def test_eval(self):
        for args, int_inputs in zip(self.good_polys["poly1_args"], self.good_polys["int_inputs"]):
            for x, correct_poly_val, correct_poly_deriv in int_inputs:
                with self.subTest():
                    self.assertEqual(Int_Polynomial(*args).eval(x, True), (correct_poly_val,correct_poly_deriv), str((args, x, correct_poly_val, correct_poly_deriv)))
        for args, int_inputs in zip(self.good_polys["poly1_args"], self.good_polys["mpf_inputs"]):
            for x, correct_poly_val, correct_poly_deriv in int_inputs:
                calc_poly_val, calc_poly_deriv = Int_Polynomial(*args).eval(x, True)
                are_almosteq = almosteq(correct_poly_val,calc_poly_val)
                if not are_almosteq:
                    with self.subTest():
                        self.assertTrue(
                            are_almosteq,
                            ("\ncalculated:  %s\n" % calc_poly_val) +
                            ("expected:    %s\n" % correct_poly_val) +
                            ("inequal_dps: %d" %inequal_dps(correct_poly_val, calc_poly_val))
                        )
                are_almosteq = almosteq(correct_poly_deriv,calc_poly_deriv)
                if not are_almosteq:
                    with self.subTest():
                        self.assertTrue(
                            are_almosteq,
                            ("\ncalculated:  %s\n" % calc_poly_deriv) +
                            ("expected:    %s\n" % correct_poly_deriv) +
                            ("inequal_dps: %d" %inequal_dps(correct_poly_deriv, calc_poly_deriv))
                        )

    def test___setitem__(self):
        for args, deg, coefs in zip(self.good_polys["poly1_args"], self.good_polys["deg"], self.good_polys["list_natural"]):
            max_deg = len(args[0]) - 1
            with self.assertRaises(IndexError):
                Int_Polynomial(*args)[max_deg+1] = 0
            with self.assertRaises(IndexError):
                Int_Polynomial(*args)[-1] = 0
            if max_deg >= 0:
                try:
                    Int_Polynomial(*args)[0] = 0
                except Exception as m:
                    self.fail("it should be possible to set the 0 coefficient. error message: %s" % m)
                try:
                    Int_Polynomial(*args)[max_deg] = 0
                except Exception as m:
                    self.fail("it should be possible to set the leading coefficient. error message: %s" % m)
                for i in range(max_deg + 1):
                    poly = Int_Polynomial(*args)
                    poly[i] = i-10
                    for k in range(deg + 1):
                        if k != i:
                            self.assertEqual(poly[k], coefs[k])
                        else:
                            self.assertEqual(poly[k], k-10)
                    for k in range(deg + 1, max_deg + 1):
                        if k != i:
                            self.assertEqual(poly[k], 0)
                        else:
                            self.assertEqual(poly[k], k-10, str((args, repr(poly), poly.get_deg(), k)))
                    self.assertEqual(poly.get_deg(), max(i, deg))
                for i,j in product(range(deg + 1),repeat=2):
                    if i != j:
                        poly = Int_Polynomial(*args)
                        poly[i] = i-10
                        poly[j] = j-12
                        for k in range(deg + 1):
                            if k != i and k != j:
                                self.assertEqual(poly[k], coefs[k])
                            elif k != i and k == j:
                                self.assertEqual(poly[k], k-12)
                            else:
                                self.assertEqual(poly[k], k-10)
                        for k in range(deg + 1, max_deg + 1):
                            if k != i and k != j:
                                self.assertEqual(poly[k], 0)
                            elif k != i and k == j:
                                self.assertEqual(poly[k], k-12)
                            else:
                                self.assertEqual(poly[k], k-10)
                        self.assertEqual(poly.get_deg(), max(i, j, deg))

    def test___getitem__(self):
        for args, deg, coefs in zip(self.good_polys["poly1_args"], self.good_polys["deg"], self.good_polys["list_natural"]):
            max_deg = len(args[0]) - 1
            if deg >= 0:
                try:
                    Int_Polynomial(*args)[max_deg+1]
                except IndexError:
                    self.fail("any positive or zero index works")
                with self.assertRaises(IndexError):
                    Int_Polynomial(*args)[-1]
            if deg >= 0:
                poly = Int_Polynomial(*args)
                for i in range(deg + 1):
                    self.assertEqual(poly[i], coefs[i])

class Test_Int_Polynomial_Array(TestCase):

    def setUp(self):
        self.several_smaller_orbits = eval_code_in_file("several_smaller_orbits.txt")
        self.arrays = []
        for _, _, Bs, _, _, _ in self.several_smaller_orbits:
            array = Int_Polynomial_Array(5, 256)
            array.init_empty(len(Bs))
            self.arrays.append(array)
            for B in Bs:
                array.append(B)

    def test_init_empty(self):
        with self.assertRaises(ValueError):
            Int_Polynomial_Array(5, 256).init_empty(-1)
        try:
            Int_Polynomial_Array(5, 256).init_empty(0)
        except ValueError:
            self.fail("Int_Polynomial_Array constructor can take init_size = 0")

    def test_append(self):
        for t, array in zip(self.several_smaller_orbits, self.arrays):
            Bs = t[2]
            with self.assertRaises(IndexError):
                array.append(Bs[-1])

    def test___len__(self):
        for _, _, Bs, _, _, _ in self.several_smaller_orbits:
            array = Int_Polynomial_Array(5, 256)
            array.init_empty(len(Bs))
            self.assertEqual(len(array), len(Bs))
            for i, B in enumerate(Bs):
                array.append(B)
                self.assertEqual(len(array), len(Bs))

    def test_get_poly(self):
        for t, array in zip(self.several_smaller_orbits, self.arrays):
            Bs = t[2]
            for i, B in enumerate(Bs):
                self.assertEqual(B, array.get_poly(i))
            with self.assertRaises(IndexError):
                array.get_poly(-1)
            with self.assertRaises(IndexError):
                array.get_poly(len(Bs))
        for _, _, Bs, _, _, _ in self.several_smaller_orbits:
            array = Int_Polynomial_Array(5, 256)
            array.init_empty(len(Bs))
            with self.assertRaises(IndexError):
                array.get_poly(0)
            for i, B in enumerate(Bs):
                array.append(B)
                with self.assertRaises(IndexError):
                    array.get_poly(i+1)

    def test_get_ndarray(self):
        for _, _, Bs, _, _, _ in self.several_smaller_orbits:
            array = Int_Polynomial_Array(5, 256)
            array.init_empty(len(Bs))
            self.assertEqual(type(array.get_ndarray()), np.ndarray)
            self.assertEqual(array.get_ndarray().shape, (0,6))
            for i, B in enumerate(Bs):
                array.append(B)
                self.assertEqual(type(array.get_ndarray()), np.ndarray)
                self.assertEqual(array.get_ndarray().shape, (i+1, 6))
                self.assertEqual(B, Int_Polynomial(array.get_ndarray()[i,:], 256))

    def test___getitem__(self):
        for t, array in zip(self.several_smaller_orbits, self.arrays):
            Bs = t[2]
            for i, B in enumerate(Bs):
                self.assertEqual(B, array[i])

        for _, _, Bs, _, _, _ in self.several_smaller_orbits:
            array = Int_Polynomial_Array(5, 256)
            array.init_empty(len(Bs))
            with self.assertRaises(IndexError):
                array[0]
            for i, B in enumerate(Bs):
                array.append(B)
                with self.assertRaises(IndexError):
                    array[i+1]

        for t, array in zip(self.several_smaller_orbits, self.arrays):
            Bs = t[2]
            starts_stops = [
                (None, None),
                (None, 1), (None, 2), (None, -2), (None, len(Bs)-2), (None, -1), (None, len(Bs)-1), (None, len(Bs)),
                (0, 1), (0, 2), (0, -2), (0, len(Bs)-2), (0, -1), (0, len(Bs)-1), (0, None), (0, len(Bs)),
                (1, 1), (1, 2), (1, -2), (1, len(Bs)-2), (1, -1), (1, len(Bs)-1), (0, None), (1, len(Bs)),
                (-2, -2), (-2, len(Bs)-2), (-2, -1), (-2, len(Bs)-1), (-2, None), (-2, len(Bs)),
                (len(Bs)-2, -2), (len(Bs)-2, len(Bs)-2), (len(Bs)-2, -1), (len(Bs)-2, len(Bs)-1), (len(Bs)-2, None), (-2, len(Bs)),
                (-1, -1), (-1, len(Bs)-1), (-1, None), (-1, len(Bs)),
                (len(Bs)-1, -1), (len(Bs)-1, len(Bs)-1), (len(Bs)-1, None), (-1, len(Bs)),
                (None, len(Bs)+1), (0, len(Bs)+1), (1, len(Bs)+1), (-2, len(Bs)+1), (-1, len(Bs)+1)
            ]
            steps = [None, 1, 2, 3, 4, 5]
            for step, (start, stop) in product(steps, starts_stops):
                subarray = array[slice(start, stop, step)]
                self.assertEqual(type(subarray), Int_Polynomial_Array)
                if start is None:
                    start_res = 0
                elif start < 0:
                    start_res = len(Bs) + start
                else:
                    start_res = start
                if stop is None:
                    stop_res = len(Bs)
                elif stop < 0:
                    stop_res = len(Bs) + stop
                elif stop > len(Bs):
                    stop_res = len(Bs)
                else:
                    stop_res = stop
                step_res = step if step else 1
                actual_len = (stop_res - start_res) // step_res
                if (stop_res - start_res) % step_res != 0:
                    actual_len += 1
                self.assertEqual(len(subarray), actual_len, "start = %s, stop = %s, step = %s" % (start, stop, step))
                for i, j in enumerate(range(start_res, stop_res, step_res)):
                    self.assertEqual(Bs[j], subarray.get_poly(i),  "start = %s, stop = %s, step = %s" % (start, stop, step))

    def test___eq__(self):
        for t, array1 in zip(self.several_smaller_orbits, self.arrays):
            Bs = t[2]
            array2 = Int_Polynomial_Array(5,32)
            array2.init_empty(len(Bs))
            for B in Bs:
                array2.append(B)
            self.assertEqual(array1, array2, "dps differing should not matter")
            array2 = copy.copy(array1)
            self.assertEqual(
                array1,
                array2,
                ("\nlen(array1) = %d\n" % len(array1)) +
                ("len(array2) = %d\n" % len(array2))
            )
            array2 = Int_Polynomial_Array(5, 256)
            array2.init_empty(len(Bs))
            for B in Bs[:-1]:
                array2.append(B)
            self.assertNotEqual(array1, array2)
            array2 = Int_Polynomial_Array(5, 256)
            array2.init_empty(len(Bs))
            for i, B in enumerate(Bs):
                if i == 0:
                    B[0] = B[0] + 1
                array2.append(B)
            self.assertNotEqual(array1, array2)

    def test_pad(self):
        for t, array1 in zip(self.several_smaller_orbits, self.arrays):
            Bs = t[2]
            array2 = copy.copy(array1)
            with self.assertRaises(ValueError):
                array2.pad(-1)
            array2 = copy.copy(array1)
            array2.pad(10)
            self.assertNotEqual(array1, array2, "\n" + str(array1.get_ndarray()) + "\n" + str(array2.get_ndarray()))
            array2.append(Bs[-1])
            self.assertNotEqual(array1, array2)
            for _ in range(9):
                array2.append(Bs[-1])
            with self.assertRaises(IndexError):
                array2.append(Bs[-1])
















