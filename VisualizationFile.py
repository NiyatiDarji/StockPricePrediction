# This class will display the results i.e. graphs
import matplotlib.pyplot as plt


class VisualizationFile:
    # Constructor
    def __init__(self):
        pass

    # Plot Graph
    @staticmethod
    def plotGraph(train_set, valid_set, predictions):
        valid_set['Predictions'] = 0
        valid_set['Predictions'] = predictions
        figure = plt.figure()
        sp = figure.add_subplot()
        sp.plot(train_set['Close'], label="Original Training Data")
        sp.plot(valid_set['Close'], label="Original Testing Data")
        sp.plot(valid_set['Predictions'], label="Predicted Testing Data")
        sp.legend(loc="best")
        return figure
