import numpy as np
class time:

    """Time class with year, month, day, hour, minute, sec and day_of_year as attributes (float)"""

    def __init__(self, date_time):
        self.year, self.month, self.day, self.hour, self.minute, self.sec = reduce_date_time(date_time, out='float')
        self.doy = doy(date_time, out = 'float')
        
def yearisLeap(year):

    """Find if variable year is a leap year. Returns True or False.
    Inputs: year: float, int or str type.
    Returns: True if year is leap | False if year is not leap.
    """

    return float(year)%4==0

def days_in_months(year):
    if yearisLeap(int(year)): 
        Feb_days=29
    else:
        Feb_days=28
    months_days = {'01':31 , '02':Feb_days, '03':31, '04':30, '05':31, '06':30, '07':31, '08':31, '09':30,
                   '10':31, '11':30, '12':31}
    return months_days

def doy(date_time, out = 'str'):

    """Convert date from YYYY-MM-DDTHH:MM:SC.### format to day of the year. If out = 'str' returns the numpy.floor(doy)
    as a string. If out is not 'str' it returns the day of year as a decimal day."""

    year, month, day, hour, minute, sec = reduce_date_time(date_time)
    months_days = days_in_months(int(year))
    hour_in_sec = float(hour)*3600.0
    min_in_sec = float(minute)*60.0
    total_sec = hour_in_sec + min_in_sec + float(sec)
    sec_in_days = total_sec/86400.0
    day_of_year =  np.nansum([days for days in months_days.values()][:int(month)-1]) + float(day) + sec_in_days
    if out == 'str':

        if day_of_year >= 100:
            day_of_year = str(int(day_of_year))
        elif day_of_year < 100 and day_of_year >= 10:
            day_of_year = '0' + str(int(day_of_year))
        else:
            day_of_year = '00' + str(int(day_of_year))

    return day_of_year

def reduce_date_time(date_time:str, out = 'str'):

    """Extract year, month, day, hour, minute, sec values from date, time in format YYYY-MM-DD-THH:MM:SC.###.
    Output is a list of the above in string type if out = 'str' (Default value), otherwise it returns a list
    of the above in float type."""

    year, month, day_time= date_time.split('-')
    day, time = day_time.split('T')
    hour, minute, sec = time.split(':')
    date_time_list = [year, month, day, hour, minute, sec]

    if out == 'str':
        return date_time_list
    
    return [float(x) for x in date_time_list]
        
def year_doy2date_str(year, doy):#, fmt='YYYY-MM-DDTHH:MM:SC.####'):
    year_str=str(int(year))
    year, month, day, hour, minutes, sec= doy2date_time(int(year), doy)
    return f'{year}-{month}-{day}T{hour}:{minutes}:{sec}'

def doy2date_time(year, doy):

    """Takes year (float) and decimal day (float) of year and returns year, month, day, hour, decimal sec as strings, to be
     used in YYYY-MM-DD-THH:MM:SC.### format, by doy2date."""
    
    months_keys = [key for key in days_in_months(year).keys()]
    
    months_days = [value for value in days_in_months(year).values()]
    
    days = doy - months_days[0]
    month_i=0
    while days>0:
        month_i += 1
        days = days - months_days[month_i]

    month = int(months_keys[month_i])

    day_of_month_dec = days + months_days[month_i]
    day_of_month = int(days + months_days[month_i])

    hour_dec = (day_of_month_dec - day_of_month)*24
    hour = int(hour_dec)

    minutes_dec = (hour_dec - hour)*60
    minutes = int(minutes_dec)

    sec = np.around((minutes_dec-minutes)*60, 4)
    if sec==60:
        if minutes+1==60:
            if hour+1==24:
                if day_of_month+1==months_days[month_i]:
                    if month+1==13:
                        year=year+1; month = '1'; day_of_month=1; hour=0; minutes=0; sec=0.0
                    else:
                        month +=1; day_of_month=1; hour=0; minutes=0; sec=0.0
                else:
                    day_of_month +=1; hour=0; minutes=0; sec=0.0
            else:
                hour +=1; minutes=0; sec=0.0
        else:
            minutes +=1; sec=0.0

    if (sec*10).is_integer(): 
        sec = str(sec)+'00'
    elif (sec*100).is_integer():
        sec = str(sec)+'0'
    else:
        sec = str(sec)
    
    return str(year), ('0'+str(month))[-2:], ('0'+str(day_of_month))[-2:], ('0'+str(hour))[-2:], ('0'+str(minutes))[-2:], ('0'+sec)[-6:]