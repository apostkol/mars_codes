from ..time_operations import time_op as tm

def mag_file_name(date_time, frame, sampl = ''):

    """Outputs the mag l2_v01_r01 file name, given the date, time, frame of reference and sampling rate."""

    day_of_year = tm.doy(date_time, out = 'str')
    year, month, day = tm.reduce_date_time(date_time)[0:3]

    name = f'mvn_mag_l2_{year}{day_of_year}{frame}{sampl}_{year}{month}{day}_v**_r**' 
    
    return name

def mag_list_files(start, end, frame, sampl):

    t = tm.time(start)
    finish = tm.time(end)
    list_file_names = []
   
    while int(t.doy) <= int(finish.doy) or t.year<finish.year:

        date_time = tm.year_doy2date_str(t.year, t.doy)
        date_str = f'{date_time}'
        list_file_names.append(mag_file_name(date_str, frame, sampl))
        
        if t.doy >= int(tm.yearisLeap(t.year))+365: 
            
            t.year +=1
            t.doy = 0

        t.doy = int(t.doy)+1

    return list_file_names
