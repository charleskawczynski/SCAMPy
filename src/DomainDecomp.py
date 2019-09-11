import sys
import copy
import numpy as np

class Domain:
    def __init__(self, N):
        self.N = N
class GridMean(Domain):
    def __init__(self, N):
        self.N = N
    def idx_name(self):
        return "gm"
class Environment(Domain):
    def __init__(self, N):
        self.N = N
    def idx_name(self):
        return "en"
class Updraft(Domain):
    def __init__(self, N):
        self.N = N
    def idx_name(self):
        if N==0:
            raise ValueError("Bad index")
        else:
            return "ud"

class DomainDecomp:
    def __init__(self, gm=0,en=0,ud=0):
        self.gm = GridMean(gm)
        self.en = Environment(en)
        self.ud = Updraft(ud)

    def sum(self):
        return self.gm.N+self.en.N+self.ud.N

    def get_domains(self):
        return self.gm,self.en,self.ud

    def get_param(self):
        return self.gm.N,self.en.N,self.ud.N
