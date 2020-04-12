import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM

x = [
    [[0], [1], [1], [0], [0], [0]],
    [[0], [0], [0], [2], [2], [0]],
    [[0], [0], [0], [0], [3], [3]],
    [[0], [2], [2], [0], [0], [0]],
    [[0], [3], [3], [3], [0], [0]],
    [[0], [0], [0], [0], [1], [1]],
]

x = np.array(x, dtype=np.float32)
y = np.array([1, 2, 3, 2, 3, 1], dtype=np.int32)

y2 = np.zeros((y.shape[0], 4), dtype=np.float32)
y2[np.arange(y.shape[0]), y] = 1.0

print(y2)


model = Sequential()
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, 1)))
model.add(Dense(4, activation='sigmoid'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(x, y2, epochs=75)

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1.25)
plt.show()


def runit(model, inp):
    inp = np.array(inp,dtype=np.float32)
    pred = model.predict(inp)
    return np.argmax(pred[0])

print( runit( model, [[[0],[2],[2],[2],[2],[0]]] ))