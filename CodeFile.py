# This class will consist of different functions for different algorithms to be applied on the data
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
import pmdarima as pm
from keras.models import Sequential
from keras.layers import Dense, LSTM


class CodeFile:

    # Constructor
    def __init__(self):
        pass

    # Pre-process the data
    @staticmethod
    def preProcessData(file_name):
        data_frame = pd.read_csv(file_name)

        # Convert date column in the format of 'y-m-d'
        data_frame['Date'] = pd.to_datetime(data_frame.Date, format='%Y-%m-%d')
        data_frame.index = data_frame['Date']
        return data_frame

    # create new data-frame with date and target variable as columns
    @staticmethod
    def __createNewDataFrame(data_frame, method):

        data = data_frame.sort_index(ascending=True, axis=0)
        new_data = pd.DataFrame(index=range(0, len(data_frame)), columns=['Date', 'Close'])

        for i in range(0, len(data)):
            new_data['Date'][i] = data['Date'][i]
            new_data['Close'][i] = data['Close'][i]

        if method != 'ma' and method != 'lstm':
            new_data['Date'] = pd.to_datetime(new_data['Date'])
            new_data['Date'] = new_data['Date'].map(dt.datetime.toordinal)

        if method == 'lstm':
            new_data.index = new_data.Date
            new_data.drop('Date', axis=1, inplace=True)
        return new_data

    # splitting into train and validation
    @staticmethod
    def __splitData(new_data, method):

        if method == 'lstm':
            dataset = new_data.values
            train_set = dataset[0:987, :]
            valid_set = dataset[987:, :]
            train_to_plot = new_data[:987]
            valid_to_plot = new_data[987:]
            return train_set, valid_set, train_to_plot, valid_to_plot
        else:
            train_set = new_data[:987]
            valid_set = new_data[987:]
            return train_set, valid_set

    # Calculate rmse value for the method
    @staticmethod
    def __rmse(y_valid, predictions):
        rmse = np.sqrt(np.mean(np.power((np.array(y_valid) - np.array(predictions)), 2)))
        return rmse

    # Moving Average Method
    @staticmethod
    def movingAverage(date_frame):

        # Create new data frame
        new_data = CodeFile().__createNewDataFrame(date_frame, "ma")
        # Seperate training set and validation set
        train_set, valid_set = CodeFile().__splitData(new_data, "ma")

        # Make predictions for validation set
        predictions = []
        for i in range(0, valid_set.shape[0]):
            a = train_set['Close'][len(train_set) - 248 + i:].sum() + sum(predictions)
            b = a / 248
            predictions.append(b)

        # Calculate Root mean square error for validation set
        y_valid = valid_set['Close']
        rmse = CodeFile().__rmse(y_valid, predictions)
        return train_set, valid_set, predictions, rmse

    # Linear Regression Method
    @staticmethod
    def linearRegression(data_frame):
        # Create new data frame
        new_data = CodeFile().__createNewDataFrame(data_frame, "lr")
        # Seperate training set and validation set
        train_set, valid_set = CodeFile().__splitData(new_data, "lr")

        x_train = train_set.drop('Close', axis=1)
        y_train = train_set['Close']
        x_valid = valid_set.drop('Close', axis=1)
        y_valid = valid_set['Close']

        # Fit to model
        model = LinearRegression()
        model.fit(x_train, y_train)

        # Make predictions and calculate the rmse
        predictions = model.predict(x_valid)
        rmse = CodeFile().__rmse(y_valid, predictions)
        return train_set, valid_set, predictions, rmse

    # K Nearest Method
    @staticmethod
    def kNearestNeighbour(data_frame):
        # Create new data frame
        new_data = CodeFile().__createNewDataFrame(data_frame, "knn")
        # Seperate training set and validation set
        train_set, valid_set = CodeFile().__splitData(new_data, "knn")

        # Scale the x-axis for training set and validation set
        scaler = MinMaxScaler(feature_range=(0, 1))

        x_train = train_set.drop('Close', axis=1)
        x_train_scaled = scaler.fit_transform(x_train)
        y_train = train_set['Close'].astype('int')
        x_valid = valid_set.drop('Close', axis=1)
        x_valid_scaled = scaler.fit_transform(x_valid)
        y_valid = valid_set['Close'].astype('int')

        # Fit to model
        model = KNeighborsClassifier(n_neighbors=3)
        model.fit(x_train_scaled, y_train)

        # Make predictions and calculate the rmse
        predictions = model.predict(x_valid_scaled)
        rmse = CodeFile().__rmse(y_valid, predictions)
        return train_set, valid_set, predictions, rmse

    # Auto ARIMA Method
    @staticmethod
    def autoArima(data_frame):
        # Create new data frame
        new_data = CodeFile().__createNewDataFrame(data_frame, "arima")
        # Seperate training set and validation set
        train_set, valid_set = CodeFile().__splitData(new_data, "arima")

        # Fit to Model
        model = pm.auto_arima(train_set['Close'], error_action='ignore', seasonal=True, suppress_warnings=True, m=12)
        model.fit(train_set['Close'])

        # Make predictions and calculate the rmse
        predictions = model.predict(n_periods=248)
        predictions = pd.DataFrame(predictions, index=valid_set.index, columns=['Prediction'])
        rmse = CodeFile().__rmse(valid_set['Close'], predictions)
        return train_set, valid_set, predictions, rmse

    # Long Short Term Memory Method
    @staticmethod
    def longShortTermMemory(data_frame):
        # Create new data frame
        new_data = CodeFile().__createNewDataFrame(data_frame, "lstm")
        # Seperate training set and validation set
        train_set, valid_set, train_to_plot, valid_to_plot = CodeFile().__splitData(new_data, "lstm")

        # Converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(new_data)

        x_train, y_train = [], []
        for i in range(60, len(train_set)):
            x_train.append(scaled_data[i - 60:i, 0])
            y_train.append(scaled_data[i, 0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        # Create and fit the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

        # Predicting 246 values, using past 60 from the train data
        inputs = new_data[len(new_data) - len(valid_set) - 60:].values
        inputs = inputs.reshape(-1, 1)
        inputs = scaler.transform(inputs)

        x_test = []
        for i in range(60, inputs.shape[0]):
            x_test.append(inputs[i - 60:i, 0])
        x_test = np.array(x_test)

        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Calculate rmse value
        rmse = CodeFile().__rmse(valid_set, predictions)
        return train_to_plot, valid_to_plot, predictions, rmse
