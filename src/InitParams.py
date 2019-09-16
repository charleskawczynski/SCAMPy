from funcs_turbulence import *

def init_params(namelist, paramlist):
    params = type('', (), {})()
    params.n_updrafts             = namelist['turbulence']['EDMF_PrognosticTKE']['updraft_number']
    params.use_local_micro        = namelist['turbulence']['EDMF_PrognosticTKE']['use_local_micro']
    params.similarity_diffusivity = namelist['turbulence']['EDMF_PrognosticTKE']['use_similarity_diffusivity']
    params.prandtl_number         = paramlist['turbulence']['prandtl_number']
    params.Ri_bulk_crit           = paramlist['turbulence']['Ri_bulk_crit']
    params.surface_area           = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']
    params.max_area_factor        = paramlist['turbulence']['EDMF_PrognosticTKE']['max_area_factor']
    params.entrainment_factor     = paramlist['turbulence']['EDMF_PrognosticTKE']['entrainment_factor']
    params.detrainment_factor     = paramlist['turbulence']['EDMF_PrognosticTKE']['detrainment_factor']
    params.pressure_buoy_coeff    = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_buoy_coeff']
    params.pressure_drag_coeff    = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_drag_coeff']
    params.pressure_plume_spacing = paramlist['turbulence']['EDMF_PrognosticTKE']['pressure_plume_spacing']
    params.tke_ed_coeff           = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_ed_coeff']
    params.tke_diss_coeff         = paramlist['turbulence']['EDMF_PrognosticTKE']['tke_diss_coeff']
    params.max_supersaturation    = paramlist['turbulence']['updraft_microphysics']['max_supersaturation']
    params.updraft_fraction       = paramlist['turbulence']['EDMF_PrognosticTKE']['surface_area']

    params.vel_pressure_coeff = params.pressure_drag_coeff/params.pressure_plume_spacing
    params.vel_buoy_coeff     = 1.0-params.pressure_buoy_coeff
    params.minimum_area       = 1e-3
    params.a_bounds           = [params.minimum_area, 1.0-params.minimum_area]
    params.w_bounds           = [0.0, 1000.0]
    params.q_bounds           = [0.0, 1.0]

    entr_src = namelist['turbulence']['EDMF_PrognosticTKE']['entrainment']
    if str(entr_src) == 'inverse_z':        params.entr_detr_fp = entr_detr_inverse_z
    elif str(entr_src) == 'dry':            params.entr_detr_fp = entr_detr_dry
    elif str(entr_src) == 'b_w2':           params.entr_detr_fp = entr_detr_b_w2
    elif str(entr_src) == 'entr_detr_tke':  params.entr_detr_fp = entr_detr_tke
    elif str(entr_src) == 'entr_detr_tke2': params.entr_detr_fp = entr_detr_tke2
    elif str(entr_src) == 'suselj':         params.entr_detr_fp = entr_detr_suselj
    elif str(entr_src) == 'none':           params.entr_detr_fp = entr_detr_none
    else: raise ValueError('Bad entr_detr_fp in Turbulence_PrognosticTKE.py')
    return params
