from Resnet import *
from matplotlib import pyplot
import numpy as np
import pickle
from scipy import stats
from sklearn.metrics import mean_squared_error
import csv

def main():

    airnow_data, forecast_data = pickle.load(open("data/my_data_o3.p", "rb"))

    start_day = 24*0
    end_day = 24*2
    testing_locations = [250010002, 250051004, 250051006, 250070001, 250092006, 250095005, 250130008, 250154002, 250170009, 250213003, 250230005, 250250042, 250270024]
    #my_location = 250010002

    airnow_data = airnow_data.reset_index(drop=True)


    for my_location in testing_locations:

        prediction_array = resnet_main(my_location)
        
        temp_forecast = forecast_data["O3"][my_location,:].reset_index(drop=True)

        graph_airnow = airnow_data[my_location][start_day:end_day]
        graph_forecast = temp_forecast[start_day:end_day]
        graph_prediction = prediction_array[start_day:end_day]
        #graph_LSTM = prediction_LSTM[start_day:end_day]


        pyplot.plot(graph_airnow, label='Airnow', color="black", linewidth=0.75)
        #pyplot.plot(graph_LSTM, label='LSTM', color="green", linewidth=0.75)
        pyplot.plot(graph_prediction, label='Resnet', color="orange", linewidth=0.75)
        pyplot.plot(graph_forecast, label='Forecast', color="blue", linewidth=0.75)
        pyplot.xlabel("Forecast Hour")
        pyplot.ylabel("Ozone (ppb)")
        pyplot.title("Station ID "+str(my_location))
        pyplot.legend()
        pyplot.grid()
        pyplot.xticks(np.arange(0, 48, 12))
        pyplot.savefig("Output/"+str(my_location)+".png")
        pyplot.clf()

        airnow_vs_prediction = mean_squared_error(graph_airnow,graph_prediction,squared=False)
        airnow_vs_forecast = mean_squared_error(graph_airnow,graph_forecast,squared=False)
        #airnow_vs_lstm = mean_squared_error(graph_airnow,graph_LSTM,squared=False)

        r_prediction, p_value1 = stats.pearsonr(graph_prediction,graph_airnow)
        r_forecast, p_value2 = stats.pearsonr(graph_forecast,graph_airnow)
        #r_lstm, p_value3 = stats.pearsonr(graph_LSTM,graph_airnow)
        print("airnow - Resnet Correlation: ", r_prediction)
        print("airnow - Forecast Correlation: ", r_forecast)

        print("airnow - Resnet RMSE: ", airnow_vs_prediction)
        print("airnow - forecast RMSE: ", airnow_vs_forecast)

        lstm_info = [
            my_location,
            airnow_vs_prediction,
            #airnow_vs_lstm,
            airnow_vs_forecast,
            r_prediction,
            #r_lstm,
            r_forecast
        ]

        with open("Output/output.csv", "a") as fp:
            wr = csv.writer(fp, dialect='excel')
            wr.writerow(lstm_info)

if __name__ == "__main__":
    main()
