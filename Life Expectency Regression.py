import sklearn
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def data_preprocess():
    data = pd.read_csv("G:/Python/Deep Learing Course/data/Life Expectancy Data.csv", header=[0])

    encodable_feature = ['Status']

    data = data.drop(['Country', 'Year'], axis=1)

    for feature in encodable_feature:
        data = encode_bind(data, feature)

    data = data.dropna()
    pre_data = data.values

    print(data.columns)

    return pre_data


def encode_bind(dataframe, feature_to_encode):
    dummies = pd.get_dummies(dataframe[[feature_to_encode]])
    result = pd.concat([dataframe, dummies], axis=1)
    result = result.drop([feature_to_encode], axis=1)
    return result


def data_split(raw_data):
    datax = np.delete(raw_data, 0, axis=1)
    datay = raw_data[:, 0]

    datay = np.reshape(datay, (-1, 1))

    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    datax = scaler_x.fit_transform(datax)
    datay = scaler_y.fit_transform(datay)

    train_data, test_data, train_labels, test_labels = train_test_split(datax, datay)

    return train_data, test_data, train_labels, test_labels


def build_model(train_dataset):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(20, activation='relu', input_shape=train_dataset.shape[1:]),
        tf.keras.layers.Dense(10, activation='softplus'),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


f = data_preprocess()
train_f, test_f, train_l, test_l = data_split(f)
model = build_model(train_f)
history = model.fit(train_f, train_l, epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)])

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 0.5)
plt.show()

predictions = model.predict(test_f).flatten()


a = plt.axes(aspect='equal')
plt.scatter(test_l, predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
limits = [-3.5, 2.25]
plt.xlim(limits)
plt.ylim(limits)
_ = plt.plot(limits, limits)
plt.show()
