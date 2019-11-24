# Main file to call functions
from CodeFile import *
from VisualizationFile import *


class Main:
    # Constructor
    def __init__(self):
        pass

    def start(self, file_name, selected_method):

        rmse = 0
        train_set = []
        valid_set = []
        predictions = []
        codefile = CodeFile()
        visualize = VisualizationFile()
        # pre-process data
        data_frame = codefile.preProcessData(file_name)

        # Moving Average method
        if selected_method == "Moving Average":
            train_set, valid_set, predictions, rmse = codefile.movingAverage(data_frame)
        elif selected_method == "Linear Regression":
            # Linear Regression method
            train_set, valid_set, predictions, rmse = codefile.linearRegression(data_frame)
        elif selected_method == "K-Nearest Neighbours":
            # K Nearest Neighbours method
            train_set, valid_set, predictions, rmse = codefile.kNearestNeighbour(data_frame)
        elif selected_method == "Auto ARIMA":
            # Auto ARIMA method
            train_set, valid_set, predictions, rmse = codefile.autoArima(data_frame)
        elif selected_method == "LSTM":
            # LSTM method
            train_set, valid_set, predictions, rmse = codefile.longShortTermMemory(data_frame)

        # noinspection PyUnboundLocalVariable
        # Visualize the data
        graph = visualize.plotGraph(train_set, valid_set, predictions)
        # Returns data, rmse value and plot to show on the GUI
        return data_frame, rmse, graph
