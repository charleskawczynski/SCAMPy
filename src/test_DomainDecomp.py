from StateVec import *
from DomainDecomp import *
from DomainSubSet import *
from DomainIdx import *
from VarMapper import *

print("---------DomainSubSet, single domain:\n")
dd,dss = DomainDecomp(gm=1),DomainSubSet(gm=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm==1
assert i_en==0
assert i_ud==(0,)
assert idx_ss==idx

dd,dss = DomainDecomp(en=1),DomainSubSet(en=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_en==1
assert i_gm==0
assert i_ud==(0,)
assert idx_ss==idx

dd,dss = DomainDecomp(ud=1),DomainSubSet(ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_en==0
assert i_gm==0
assert i_ud==(1,)
assert idx_ss==idx

dd,dss = DomainDecomp(ud=2),DomainSubSet(ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_en==0
assert i_gm==0
assert i_ud==(1,2)
assert idx_ss==idx

print("---------DomainSubSet, two domains:\n")
dd,dss = DomainDecomp(gm=1,en=1),DomainSubSet(gm=True,en=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm==2
assert i_en==1
assert i_ud==(0,)
assert idx_ss==idx

dd,dss = DomainDecomp(gm=1,ud=3),DomainSubSet(gm=True,ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm==4
assert i_en==0
assert i_ud==(1,2,3)
assert idx_ss==idx

dd,dss = DomainDecomp(en=1,ud=3),DomainSubSet(en=True,ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm==0
assert i_en==4
assert i_ud==(1,2,3)
assert idx_ss==idx

dd,dss = DomainDecomp(gm=1,ud=1),DomainSubSet(gm=True,ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm==2
assert i_en==0
assert i_ud==(1,)
assert idx_ss==idx

dd,dss = DomainDecomp(en=1,ud=1),DomainSubSet(en=True,ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm==0
assert i_en==2
assert i_ud==(1,)
assert idx_ss==idx

print("---------DomainIdx, all domains:\n")
dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=True,en=True,ud=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm == 6
assert i_en == 5
assert i_ud == (1,2,3,4)
assert idx_ss==idx

print("---------DomainIdx, utilizing DomainSubSet:\n")
dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=True)
idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()
assert i_gm == 6
assert i_en == 5
assert i_ud == (1,2,3,4)

i_gm, i_en, i_ud, i_sd, i_al = idx_ss.allcombinations()
assert i_gm == 1
assert i_en == 0
assert i_ud == (0,)

print("---------Test DomainSubSet indexing, multiple domains:\n")
dd, dss = DomainDecomp(gm=1,en=1,ud=4), DomainSubSet(gm=True)

idx = DomainIdx(dd)
idx_ss = DomainIdx(dd, dss)
i_gm, i_en, i_ud, i_sd, i_al = idx.allcombinations()

idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (0,)
assert idx_ss.environment() == 0
assert idx_ss.gridmean() == 1

dss = DomainSubSet(en=True)
idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (0,)
assert idx_ss.environment() == 1
assert idx_ss.gridmean() == 0

dss = DomainSubSet(ud=True)
idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (1,2,3,4)
assert idx_ss.environment() == 0
assert idx_ss.gridmean() == 0

dss = DomainSubSet(gm=True,en=True)
idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (0,)
assert idx_ss.environment() == 1
assert idx_ss.gridmean() == 2

dss = DomainSubSet(gm=True,ud=True)
idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (1,2,3,4)
assert idx_ss.environment() == 0
assert idx_ss.gridmean() == 5

dss = DomainSubSet(en=True,ud=True)
idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (1,2,3,4)
assert idx_ss.environment() == 5
assert idx_ss.gridmean() == 0

dss = DomainSubSet(gm=True,en=True,ud=True)
idx_ss = DomainIdx(dd, dss)
assert idx_ss.updraft() == (1,2,3,4)
assert idx_ss.environment() == 5
assert idx_ss.gridmean() == 6

print("---------Test global indexing:\n")

var_set = (('ρ_0', DomainSubSet(gm=True),                 None, None),
           ('a',   DomainSubSet(gm=True,en=True,ud=True), None, None),
           ('tke', DomainSubSet(en=True,ud=True),         None, None),
           ('K_h', DomainSubSet(ud=True),                 None, None))

dd = DomainDecomp(gm=1,en=1,ud=4)
idx = DomainIdx(dd)

vm, dss_per_var, var_names = get_var_mapper(var_set, dd)

idx_ss_per_var = {name : DomainIdx(dd, dss_per_var[name]) for name in var_names}
a_map = {name : get_sv_a_map(idx, idx_ss_per_var[name]) for name in var_names}
i_gm,i_en,i_ud = idx.eachdomain()

assert get_i_state_vec(vm, a_map['ρ_0'], 'ρ_0', i_gm) == 0
assert get_i_state_vec(vm, a_map['a']  , 'a', i_ud[0]) == 1
assert get_i_state_vec(vm, a_map['a']  , 'a', i_ud[1]) == 2
assert get_i_state_vec(vm, a_map['a']  , 'a', i_ud[2]) == 3
assert get_i_state_vec(vm, a_map['a']  , 'a', i_ud[3]) == 4
assert get_i_state_vec(vm, a_map['a']  , 'a', i_en)    == 5
assert get_i_state_vec(vm, a_map['a']  , 'a', i_gm)    == 6
assert get_i_state_vec(vm, a_map['tke'], 'tke', i_ud[0]) == 7
assert get_i_state_vec(vm, a_map['tke'], 'tke', i_ud[1]) == 8
assert get_i_state_vec(vm, a_map['tke'], 'tke', i_ud[2]) == 9
assert get_i_state_vec(vm, a_map['tke'], 'tke', i_ud[3]) == 10
assert get_i_state_vec(vm, a_map['tke'], 'tke', i_en)    == 11
assert get_i_state_vec(vm, a_map['K_h'], 'K_h', i_ud[0]) == 12
assert get_i_state_vec(vm, a_map['K_h'], 'K_h', i_ud[1]) == 13
assert get_i_state_vec(vm, a_map['K_h'], 'K_h', i_ud[2]) == 14
assert get_i_state_vec(vm, a_map['K_h'], 'K_h', i_ud[3]) == 15

