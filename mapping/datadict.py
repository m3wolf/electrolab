# -*- coding: utf-8 -*-

class DataDict():
    """Provides a convenient way to get a dictionary of pre-computed values."""
    def __init__(self, attrs=[]):
        self.attrs = attrs

    def __get__(self, obj, cls):
        dataDict = {}
        for attr in self.attrs:
            dataDict[attr] = getattr(obj, attr, None)
        return dataDict

    def __set__(self, obj, newDict):
        if newDict is not None:
            for attr in newDict.keys():
                setattr(obj, attr, newDict[attr])
