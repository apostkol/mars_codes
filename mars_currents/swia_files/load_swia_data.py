from mars_currents.swia_files import swia_file_names as sfn 
import glob
from mars_currents.spice_codes.load_spice_kernels import load_spice_kernels
import cdflib
from cdflib.xarray import cdf_to_xarray
import xarray as xr
import warnings 
def create_paths(start, end, kind, data_path, verbose=True):

    """Returns a list of paths for the swia files. """

    paths = []
    not_exist = []
    for name in sfn.swia_list_files(start, end, kind):
                
        name_st = 'mvn_swi_l2_'
        year = name.replace(f'{name_st}{kind}_', '')[:4]
        month = name.replace(f'{name_st}{kind}_{year}', '')[:2]
        day = name.replace(f'{name_st}{kind}_{year}{month}', '')[:2]  
        file_path = f'{data_path}/{year}/{month}/{name}.cdf'
        file_in_path = glob.glob(file_path)
        
        if not bool(file_in_path):
            not_exist.append(file_path)            
        elif len(file_in_path)>1:
            paths.append(file_in_path[-1])
        else:
            paths.append(file_in_path[0])
            
    if verbose:
        print('Paths of files: \n', paths)
        
    return [paths, not_exist]

def load_data(start, end, kind, data_path, kernel_path, resampling = None, verbose=True, thin_n = None):

    load_spice_kernels(kernel_path)
    paths, not_exist = create_paths(start, end, kind, data_path, False)
    
    if verbose:
        print('These files do not exist: \n')
        for no_file in not_exist:
            print(f'{no_file}')
        print('--------------------------------------------------------------------')
    
    if verbose: print('Loading files: \n')
    
    for path in paths:
        if verbose: print("\r", path, end =" ")
        if path==paths[0]:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if resampling is not None:
                    xar0 = cdf_to_xarray(path, to_unixtime=False, to_datetime=True)
                else:
                    xar0 = cdf_to_xarray(path, to_unixtime=False, to_datetime=False)

                
                if resampling is not None:
                    
                    if resampling['mode'] == 'median':
                        
                        xar0 = xar0.resample(epoch=resampling['res_freq']).median(dim ='epoch')
                        
                    else:
                        
                        xar0 = xar0.resample(epoch=resampling['res_freq']).mean(dim ='epoch')
                    
                    
                elif thin_n is not None:
                    
                    xar0 = xar0.thin(indexers = dict(epoch=thin_n))
                    
            if xar0['epoch'].attrs['units']=='ns':
                
                xar0['epoch'].attrs['units'] = 's'; xar0['epoch'].attrs['UNITS'] = 's';
                old_xar_attrs = xar0['epoch'].attrs
                new_t = xar0['epoch'].values*1e-9
                xar0 = xar0.assign_coords(epoch = new_t)
                xar0['epoch'].attrs.update(old_xar_attrs)
                
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                if resampling is not None:
                    xar1 = cdf_to_xarray(path, to_unixtime=False, to_datetime=True)
                else:
                    xar1 = cdf_to_xarray(path, to_unixtime=False, to_datetime=False)
                    
                if resampling is not None:
                    
                    if resampling['mode'] == 'median':

                        xar1 = xar1.resample(epoch=resampling['res_freq']).median(dim ='epoch')

                    else:

                        xar1 = xar1.resample(epoch=resampling['res_freq']).mean(dim ='epoch')

                elif thin_n is not None:
                    
                    xar1 = xar1.thin(indexers = dict(epoch=thin_n))
                    
            if xar1.epoch[0] < xar0.epoch[-1]:
                
                xar1 = xar1.where(xar1.epoch>xar0.epoch[-1], drop = True)
            
            if xar1['epoch'].attrs['units']=='ns':
                xar1['epoch'].attrs['units'] = 's'; xar1['epoch'].attrs['UNITS'] = 's';
                old_xar_attrs = xar1['epoch'].attrs
                new_t = xar1['epoch'].values*1e-9
                xar1 = xar1.assign_coords(epoch = new_t)
                xar1['epoch'].attrs.update(old_xar_attrs)
            
            xar0 = xr.combine_by_coords(data_objects = [xar0, xar1], combine_attrs='override')#, compat = 'override',)# coords= 'all')
    
    return xar0