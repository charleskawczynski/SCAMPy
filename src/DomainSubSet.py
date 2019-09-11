import sys
import copy
import numpy as np
from DomainDecomp import *

class DomainSubSet:
    def __init__(self, gm=False,en=False,ud=False):
        self.gm = GridMean(gm)
        self.en = Environment(en)
        self.ud = Updraft(ud)

    def gridmean(self, dd=None):
        if dd==None:
            return self.gm.N
        else:
            return dd.gm.N if self.gm.N else 0
    def environment(self, dd=None):
        if dd==None:
            return self.en.N
        else:
            return dd.en.N if self.en.N else 0
    def updraft(self, dd=None):
        if dd==None:
            return self.ud.N
        else:
            return dd.ud.N if self.ud.N else 0

    def sum(self, dd):
        return self.gridmean(dd)+self.environment(dd)+self.updraft(dd)

    def get_param(self, dd):
        return self.gridmean(dd),self.environment(dd),self.updraft(dd)
