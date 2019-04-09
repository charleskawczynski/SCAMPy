import netCDF4 as nc
import os
import shutil

from TimeStepping import TimeStepping

from Grid import Grid
import numpy as np

class NetCDFIO_Stats:
    def __init__(self, namelist, paramlist, Gr):
        self.root_grp = None
        self.profiles_grp = None
        self.ts_grp = None
        self.Gr = Gr

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])

        self.frequency = namelist['stats_io']['frequency']

        # Setup the statistics output path
        outpath = str(os.path.join(namelist['output']['output_root'] + 'Output.' + namelist['meta']['simname'] + '.'
                                   + self.uuid[len(self.uuid )-5:len(self.uuid)]))

        try:
            os.mkdir(outpath)
        except:
            pass

        self.stats_path = str( os.path.join(outpath, namelist['stats_io']['stats_dir']))

        try:
            os.mkdir(self.stats_path)
        except:
            pass


        self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname'] + '.nc')
        if os.path.exists(self.path_plus_file):
            for i in range(100):
                res_name = 'Restart_'+str(i)
                print("Here " + res_name)
                if os.path.exists(self.path_plus_file):
                    self.path_plus_file = str( self.stats_path + '/' + 'Stats.' + namelist['meta']['simname']
                           + '.' + res_name + '.nc')
                else:
                    break



        shutil.copyfile(os.path.join( './', namelist['meta']['simname'] + '.in'),
                        os.path.join( outpath, namelist['meta']['simname'] + '.in'))

        shutil.copyfile(os.path.join( './paramlist_'+paramlist['meta']['casename']+ '.in'),
                        os.path.join( outpath, 'paramlist_'+paramlist['meta']['casename']+ '.in'))
        self.setup_stats_file()
        return

    def open_files(self):
        self.root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        self.profiles_grp = self.root_grp.groups['profiles']
        self.ts_grp = self.root_grp.groups['timeseries']
        return

    def close_files(self):
        self.root_grp.close()
        return

    def setup_stats_file(self):
        kmin = self.Gr.gw
        kmax = self.Gr.nzg-self.Gr.gw

        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

        # Set profile dimensions
        profile_grp = root_grp.createGroup('profiles')
        profile_grp.createDimension('z', self.Gr.nz)
        profile_grp.createDimension('t', None)
        z = profile_grp.createVariable('z', 'f8', ('z'))
        z[:] = np.array(self.Gr.z[kmin:kmax])
        z_half = profile_grp.createVariable('z_half', 'f8', ('z'))
        z_half[:] = np.array(self.Gr.z_half[kmin:kmax])
        profile_grp.createVariable('t', 'f8', ('t'))
        del z
        del z_half

        reference_grp = root_grp.createGroup('reference')
        reference_grp.createDimension('z', self.Gr.nz)
        z = reference_grp.createVariable('z', 'f8', ('z'))
        z[:] = np.array(self.Gr.z[kmin:kmax])
        z_half = reference_grp.createVariable('z_half', 'f8', ('z'))
        z_half[:] = np.array(self.Gr.z_half[kmin:kmax])
        del z
        del z_half

        ts_grp = root_grp.createGroup('timeseries')
        ts_grp.createDimension('t', None)
        ts_grp.createVariable('t', 'f8', ('t'))

        root_grp.close()
        return

    def add_profile(self, var_name):

        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        profile_grp = root_grp.groups['profiles']
        new_var = profile_grp.createVariable(var_name, 'f8', ('t', 'z'))

        root_grp.close()

        return

    def add_reference_profile(self, var_name):
        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        reference_grp = root_grp.groups['reference']
        new_var = reference_grp.createVariable(var_name, 'f8', ('z',))
        root_grp.close()

        return

    def add_ts(self, var_name):

        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        ts_grp = root_grp.groups['timeseries']
        new_var = ts_grp.createVariable(var_name, 'f8', ('t',))

        root_grp.close()
        return

    def write_profile(self, var_name, data):
        var = self.profiles_grp.variables[var_name]
        var[-1, :] = np.array(data)
        return

    def write_reference_profile(self, var_name, data):
        '''
        Writes a profile to the reference group NetCDF Stats file. The variable must have already been
        added to the NetCDF file using add_reference_profile
        :param var_name: name of variables
        :param data: data to be written to file
        :return:
        '''

        root_grp = nc.Dataset(self.path_plus_file, 'r+', format='NETCDF4')
        reference_grp = root_grp.groups['reference']
        var = reference_grp.variables[var_name]
        var[:] = np.array(data)
        root_grp.close()
        return

    def write_ts(self, var_name, data):
        var = self.ts_grp.variables[var_name]
        var[-1] = data
        return

    def write_simulation_time(self, t):
        # Write to profiles group
        profile_t = self.profiles_grp.variables['t']
        profile_t[profile_t.shape[0]] = t

        # Write to timeseries group
        ts_t = self.ts_grp.variables['t']
        ts_t[ts_t.shape[0]] = t

        return

