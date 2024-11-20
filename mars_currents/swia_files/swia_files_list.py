from ..time_operations import time_op as tm

def _swia_file_name(date_time, kind):

    """Outputs the swia file name, given the date, time, type.
    Inputs:

    date_time (str): date and time in YYY-MM-DDTHH:MM:SC.#### format
    type (str): Either of coarsesvy3d, coarsearc3d, finesvy3d, finearc3d, onboardsvymom, onboardsvyspec

    Ouputs:

    swia data file name in format (str): mvn_swi_l2_<type>_<yyyy><mm><dd>_v<xx>_r<xx>
    """

    day_of_year = tm.doy(date_time, out = 'str')
    year, month, day = tm.reduce_date_time(date_time)[0:3]

    name = f'mvn_swi_l2_{kind}_{year}{month}{day}_v**_r**'
    
    return name

def swia_files_list(start, end, kind):
    t = tm.time(start)
    finish = tm.time(end)
    list_file_names = []

    while int(t.doy) <= int(finish.doy) or t.year<finish.year:

        date_time = tm.year_doy2date_str(t.year, t.doy)
        date_str = f'{date_time}'
        list_file_names.append(_swia_file_name(date_str, kind))
        
        if t.doy >= int(tm.yearisLeap(t.year))+365: 
            
            t.year +=1
            t.doy = 0

        t.doy =t.doy+1

    return list_file_names

