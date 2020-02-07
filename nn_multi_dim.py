import sys
import matplotlib.pyplot as plt
import numpy as np
import coremltools
from numpy import loadtxt
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler


training_file = sys.argv[1]
testing_file = sys.argv[2]

dataset = loadtxt(str(training_file), delimiter=',')
x = dataset[:,1:5]
y = dataset[:,0]
y = np.reshape(y, (-1,1))

model = Sequential()
model.add(Dense(20, input_dim=4, kernel_initializer='normal', activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(x, y, epochs=100, batch_size=10,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model.save("angleModel_multi_dim.h5")
print("Saved model to the disk")
