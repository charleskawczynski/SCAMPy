import netCDF4 as nc
import os
import shutil

from TimeStepping import TimeStepping

from Grid import Grid, Zmin, Zmax, Center, Node
import numpy as np

class NetCDFIO_Stats:
    def __init__(self, namelist, paramlist, Gr, root_dir):
        self.root_grp = None
        self.profiles_grp = None
        self.ts_grp = None
        self.grid = Gr

        self.last_output_time = 0.0
        self.uuid = str(namelist['meta']['uuid'])

        self.frequency = namelist['stats_io']['frequency']

        # Setup the statistics output path
        p = []
        p.append(namelist['output']['output_root'])
        p.append('Output.')
        p.append(namelist['meta']['simname'])
        p.append('.')
        p.append(self.uuid[len(self.uuid )-5:len(self.uuid)])
        self.outpath = str(os.path.join(root_dir, ''.join(p))) + os.sep
        self.figpath = str(os.path.join(root_dir, ''.join(p))) + os.sep + 'figs'+os.sep

        os.makedirs(self.outpath, exist_ok=True)
        os.makedirs(self.figpath, exist_ok=True)

        self.stats_path = str(os.path.join(self.outpath, namelist['stats_io']['stats_dir']))

        os.makedirs(self.stats_path, exist_ok=True)

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

        shutil.move(os.path.join( './', namelist['meta']['simname'] + '.in'),
                    os.path.join( self.outpath, namelist['meta']['simname'] + '.in'))

        shutil.move(os.path.join( './paramlist_'+paramlist['meta']['casename']+ '.in'),
                    os.path.join( self.outpath, 'paramlist_'+paramlist['meta']['casename']+ '.in'))
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
        k_b_1 = self.grid.boundary(Zmin())
        k_b_2 = self.grid.boundary(Zmax())
        k_1 = k_b_1
        k_2 = k_b_2
        # k_1 = self.grid.first_interior(Zmin()) # IO assumes full and half fields are equal sizes
        # k_2 = self.grid.first_interior(Zmax()) # IO assumes full and half fields are equal sizes

        root_grp = nc.Dataset(self.path_plus_file, 'w', format='NETCDF4')

        # Set profile dimensions
        profile_grp = root_grp.createGroup('profiles')
        profile_grp.createDimension('z', self.grid.nz)
        profile_grp.createDimension('t', None)
        z = profile_grp.createVariable('z', 'f8', ('z'))
        z[:] = np.array(self.grid.z[k_b_1:k_b_2])
        z_half = profile_grp.createVariable('z_half', 'f8', ('z'))
        z_half[:] = np.array(self.grid.z_half[k_1:k_2])
        profile_grp.createVariable('t', 'f8', ('t'))
        del z
        del z_half

        reference_grp = root_grp.createGroup('reference')
        reference_grp.createDimension('z', self.grid.nz)
        z = reference_grp.createVariable('z', 'f8', ('z'))
        z[:] = np.array(self.grid.z[k_b_1:k_b_2])
        z_half = reference_grp.createVariable('z_half', 'f8', ('z'))
        z_half[:] = np.array(self.grid.z_half[k_1:k_2])
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

    def write_profile_new(self, var_name, grid, data):
        var = self.profiles_grp.variables[var_name]
        k_1 = grid.boundary(Zmin())
        k_2 = grid.boundary(Zmax())
        var[-1, :] = np.array(data[k_1:k_2])
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

