import sys
import copy
import numpy as np
from DomainDecomp import *
from DomainSubSet import *

def get_i_state_vec(vm, a_map, name, i_sd=0):
    # print('---------- get_i_state_vec')
    i_a = i_sd-1
    # print('name = ', name)
    # print('i_a = ', i_a)
    # print('a_map = ', a_map)
    # print('a_map[i_a] = ', a_map[i_a])
    # print('vm[name] = ', vm[name])
    return vm[name][a_map[i_a]]

def get_idx(gm,en,ud):
    i_gm,i_en,i_ud = 0,0,(0,)
    if ud>0:
        i_ud = tuple([i+1 for i in range(ud)])
    if en>0:
        i_en = max(i_ud)+1
    if gm>0 and en>0:
        i_gm = i_en+1
    if gm>0 and not en>0:
        i_gm = max(i_ud)+1
    return i_gm,i_en,i_ud

class DomainIdx:
    def __init__(self, dd, dss=None):
        gm,en,ud = dd.get_param() if dss==None else dss.get_param(dd)
        self.i_gm, self.i_en, self.i_ud = get_idx(gm,en,ud)

    def __str__(self,):
        s = ''
        s+='i_gm = '+str(self.i_gm)+', '
        s+='i_en = '+str(self.i_en)+', '
        s+='i_ud = '+str(self.i_ud)
        return s

    def __eq__(self, other):
        T1 = self.i_gm == other.i_gm
        T2 = self.i_en == other.i_en
        T3 = all([x==y for (x,y) in zip(self.i_ud,other.i_ud)])
        L = all([T1,T2,T3])
        if not L:
            print('self  = ',self)
            print('other = ',other)
        return L

    def gridmean(self, dd=None):
        return self.i_gm
    def environment(self, dd=None):
        return self.i_en
    def updraft(self, dd=None):
        return self.i_ud

    def has_gridmean(self):
        return not self.i_gm == 0
    def has_environment(self):
        return not self.i_en == 0
    def has_updraft(self):
        return not self.i_ud == (0,)

    # Return flat list
    def subdomains(self):
        return tuple([self.environment()]+list(self.updraft()))
    def alldomains(self):
        return tuple([self.gridmean()]+[self.environment()]+list(self.updraft()))

    # Return structured
    def eachdomain(self):
        return self.gridmean(),self.environment(),self.updraft()
    def allcombinations(self):
        return self.gridmean(),self.environment(),self.updraft(),self.subdomains(),self.alldomains()

    def var_suffix(self, idx, name, i_sd=0):
        if i_sd == idx.gridmean():
            return "_gm"
        elif i_sd == idx.environment():
            return "_en"
        elif i_sd in idx.updraft():
            return "_ud_"+str(i_sd)
        else:
            raise ValueError("Bad index in var_suffix")

