import numpy as np

def get_var_mapper(var_tuple, dd):
    var_names = [var_name for (var_name, dss, loc, bc) in var_tuple]
    n_sd_per_var = [dss.sum(dd) for (var_name, dss, loc, bc) in var_tuple]
    end_index  = np.cumsum(n_sd_per_var)

    var_names_test = sorted(var_names)
    var_names_unique = sorted(list(set(var_names_test)))
    assert len(var_names_test)==len(var_names_unique)
    assert all([x==y for x,y in zip(var_names_test, var_names_unique)])

    start_index = [0]+[x for x in end_index][0:-1]
    vals = [list(range(a,b)) for a,b in zip(start_index, end_index)]
    var_mapper = {k : v for k,v in zip(var_names, vals)}
    dss_per_var = {var_name : dss for (var_name, dss, loc, bc) in var_tuple}
    return var_mapper, dss_per_var, var_names

def get_sv_a_map(idx, idx_ss):
    a = np.zeros(len(idx.alldomains()), dtype=int)
    for i in idx.alldomains():
        if i == idx.gridmean() and idx_ss.has_gridmean():
            a[i-1] = int(idx_ss.gridmean()-1)
        elif i == idx.environment() and idx_ss.has_environment():
            a[i-1] = int(idx_ss.environment()-1)
        elif i in idx.updraft() and idx_ss.has_updraft():
            a[i-1] = int(idx_ss.updraft()[i-1]-1)
        else:
            a[i-1] = int(-1)
    return a

def get_sd_unmapped(var_tuple, idx, dd):
    sd_unmapped = {}
    i_all = idx.alldomains()
    for v,dss in var_tuple:
        idx_ss = DomainIdx(dd, dss)
        if not v in sd_unmapped:
            sd_unmapped[v] = []
        for i in i_all:
            if i == idx_ss.gridmean():
                sd_unmapped[v] += [idx.gridmean()]
            if i == idx_ss.environment():
                sd_unmapped[v] += [idx.environment()]
            if i in idx_ss.updraft() and not i in sd_unmapped[v]:
                for k in idx.updraft():
                    sd_unmapped[v] = [k]
    return sd_unmapped
