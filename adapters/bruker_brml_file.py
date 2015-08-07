# -*- coding: utf-8 -*-

from xml.etree import ElementTree
import zipfile

import pandas as pd

class BrukerBrmlFile():
    def __init__(self, filename):
        with zipfile.ZipFile(filename) as zf:
            dataFile = zf.open('Experiment0/RawData0.xml')
            self._dataTree = ElementTree.parse(dataFile)

    @property
    def sample_name(self):
        nameElement = self._dataTree.find('.//InfoItem[@Name="SampleName"]')
        name = nameElement.get('Value')
        return name

    @property
    def dataframe(self):
        index = []
        countsList = []
        # Find all Datum entries in data tree
        data = self._dataTree.findall('.//Datum')
        for datum in data:
            time, num, two_theta, theta, counts = datum.text.split(',')
            index.append(float(two_theta))
            countsList.append(int(counts))
        # Build pandas DataFrame
        df = pd.DataFrame(countsList, index=index, columns=['counts'])
        df.index.name = 'two_theta'
        return df
