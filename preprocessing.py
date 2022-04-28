import pickle
import pandas as pd
import numpy as np

#13 - 23 inc missing on 2019-12-31
#maybe add 0 rows and then interpolate
#This is true in the original data too

def preprocessing():
    

    # Load the data
    airnow_data, forecast_data = pickle.load(open("data/data_o3.p", "rb"))

    ##  Modifying Airnow Data
    ##

    #Keys are station ID
    for item in airnow_data.keys():
        #Keys are dates
        for date in airnow_data[item].keys():

            # Marking missing data in airnow
            if airnow_data[item][date] == -999000:
                airnow_data[item][date] = None

        #Interpolating missing data
        airnow_data[item].interpolate(method="time",inplace=True,limit_direction="both")

    ##  Modifying Forecast Data
    ##
    forecast_data.index = forecast_data.index.droplevel(0)
    forecast_data = forecast_data[~forecast_data.index.duplicated(keep='first')]


    missing_days = ["2019-12-31 13:00:00","2019-12-31 14:00:00","2019-12-31 15:00:00","2019-12-31 16:00:00","2019-12-31 17:00:00",
    "2019-12-31 18:00:00","2019-12-31 19:00:00","2019-12-31 20:00:00","2019-12-31 21:00:00","2019-12-31 22:00:00",
    "2019-12-31 23:00:00"]
    missing_days_nums = [13,14,15,16,17,18,19,20,21,22,23]
    cols = ['NOX', 'NOY', 'O3', 'PBL2', 'RGRND', 'TEMP2', 'TIMEOFDAY', 'WDIR10','WSPD10']


    weird_array = []
    indexes = [[],[]]
    for i in range(len(missing_days)):
        for station in airnow_data.keys():
            indexes[0].append(station)
            indexes[1].append(missing_days[i])
            append_this = [missing_days_nums[i]]*9
            weird_array.append(append_this)


    tuples = list(zip(*indexes))
    index = pd.MultiIndex.from_tuples(tuples, names=["stationid", "date_time"])
    df2 = pd.DataFrame(weird_array,columns=cols,index=index)
    forecast_data = forecast_data.append(df2)

    """    for item in forecast_data.keys():
        print("here")
        #Keys are dates
        for date in forecast_data[item].keys():
            if math.isnan(forecast_data[item][date]):
                forecast_data[item][date] = None

        forecast_data[item] = forecast_data[item].interpolate(method="linear")"""


    dbfile = open('data/my_data_o3.p', 'wb')
      
    # source, destination
    pickle.dump((airnow_data,forecast_data), dbfile)                     
    dbfile.close()

preprocessing()