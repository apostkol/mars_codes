import numpy as np 
import glob
from .mag_file_names import mag_list_files
import os
import spiceypy as spice
#from spiceypy.utils.support_types import SpiceyError
from ..spice_codes.load_spice_kernels import load_spice_kernels

def create_paths(start, end, frame, sampl, data_path, verbose=True):
    
    """Returns a list of paths for the mag files. """
    
    paths = []
    not_exist = []
    for name in mag_list_files(start, end, frame, sampl):
        
        year = name[11:15]; month = name[-12:-10]
        
        file_path = f'{data_path}/{year}/{month}/{name}.sts'
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

def load_b(start, end, frame, sampl, data_path, kernel_path, verbose=True, re_sampling = None):
    
    load_spice_kernels(os.path.normpath(kernel_path))
    paths, not_exist = create_paths(start, end, frame, sampl, data_path, verbose = False)
    
    if verbose:
        print('These files do not exist: \n')
        for no_file in not_exist:
            print(f'{no_file}')
        print('--------------------------------------------------------------------')
    output = {'time':None}
    
    if verbose: print('Loading files: \n')
    
    for name in paths:
        if verbose: print("\r", name, end =" ")
        
        # header size is variable. Find four END_OBJECT in a row
        count = 0
        skip = 1

        with open(name,'r') as f:
            
            for line in f.readlines():
                
                skip += 1
                
                if line.strip() == 'END_OBJECT':
                    
                    count += 1
                
                else:
                
                    count = 0
                
                if count == 3:
                    
                    break
            else:
                
                raise ValueError("Could not parse the header in file %s" % f)
            
        year = name.split('_')[-3][:4] + '-001T00:00:00.000'
        
        if re_sampling is not None:
            
            c = []
            
            with open(name, 'r') as f:
                
                for line_count, line in enumerate(f.readlines()[skip:]):
                    
                    if line_count % re_sampling == 0:
                        
                        line_array = np.fromstring(line, sep = " ").tolist()

                        c.append(line_array[6:10] + line_array[11:14])
                    
            c = np.stack(c, axis = 1)


        else:
            
            c = np.loadtxt(name, skiprows=skip, usecols=[6,7,8,9,11,12,13]).T
                
        c[0] = (c[0]-1.0)*86400. + spice.utc2et(year)

        if name == paths[0]:
            
            output['b'] = np.array(c[1:4])
            output['time'] = np.array(c[0])
            output['position'] = np.array(c[4:])
        
        else:
            
            output['b'] = np.hstack((output['b'], np.array(c[1:4])))
            output['time'] = np.hstack((output['time'],np.array(c[0])))
            output['position'] = np.hstack((output['position'], np.array(c[4:])))
            
    return output