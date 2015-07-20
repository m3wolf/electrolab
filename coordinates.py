# -*- coding: utf-8 -*-

import math

class Cube():
    """Cubic coordinates of a hexagon"""
    @staticmethod
    def from_xy(xy, unit_size):
        x, y = (xy[0], xy[1])
        j = (y/math.sqrt(3)-x)/unit_size
        i = 2*y/math.sqrt(3)/unit_size - j
        i = round(i)
        j = round(j)
        return Cube(i, j, -(i+j))

    def __init__(self, i, j, k, *args, **kwargs):
        self.i = i
        self.j = j
        self.k = k

    def __getitem__(self, key):
        coord_list = [self.i, self.j, self.k]
        return coord_list[key]

    def __add__(self, other):
        new = Cube(
            self.i + other.i,
            self.j + other.j,
            self.k + other.k,
        )
        return new

    def __eq__(self, other):
        result = False
        if self.i == other.i and self.j == other.j and self.k == other.k:
            result = True
        return result

    def __str__(self):
        return "({i}, {j}, {k})".format(i=self.i, j=self.j, k=self.k)

    def __repr__(self):
        return "Cube{0}".format(self.__str__())
