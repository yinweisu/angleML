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
widthHeightRatio = dataset[:,0]
widthHeightRatio = np.reshape(widthHeightRatio, (-1,1))
pitch = dataset[:,1]
pitch = np.reshape(pitch, (-1,1))
scalar_x = MinMaxScaler()
scalar_y = MinMaxScaler()
print(scalar_x.fit(widthHeightRatio))
xscale = scalar_x.transform(widthHeightRatio)
print(scalar_y.fit(pitch))
yscale = scalar_y.transform(pitch)

# x_train, x_test, y_train, y_test = train_test_split(xscale, yscale)

model = Sequential()
model.add(Dense(10, input_dim=1, kernel_initializer='normal', activation='tanh'))
# model.add(Dense(10, activation='sigmoid'))
model.add(Dense(1, activation='tanh'))
model.summary()
opt = optimizers.SGD(lr=0.01, momentum=0.9)
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])

history = model.fit(xscale, yscale, epochs=100, batch_size=5,  verbose=1, validation_split=0.2)

print(history.history.keys())
# "Loss"
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')

test_dataset = loadtxt(str(testing_file), delimiter=',')
xnew = test_dataset[:,0]
yreal = test_dataset[:,1]
xnew = np.reshape(xnew, (-1,1))
xnew = scalar_x.transform(xnew)
ynew = model.predict(xnew)

ynew = scalar_y.inverse_transform(ynew)
xnew = scalar_x.inverse_transform(xnew)
mae = sum([abs(y_predict-y) for (y_predict, y) in zip(ynew, yreal)]) / len(ynew)
print(mae)
area = np.pi*3

plt.figure(2)
plt.scatter(xnew, yreal, s=area, alpha=0.5)
plt.scatter(xnew, ynew, s=area, alpha=0.5)
plt.show()

model.save("angleModel.h5")
print("Saved model to the disk")

# def createModel():
#   model = Sequential()
#   model.add(Dense(1, input_dim=1, activation='tanh'))
#   model.add(Dense(1,  kernel_initializer='normal'))
#   model.compile(loss='mean_squared_error', optimizer=opt)
#   return model
#
# estimator = KerasRegressor(build_fn=createModel, epochs=100, batch_size=32, verbose=1)
# kfold = KFold(n_splits=10)
# results = cross_val_score(estimator, widthHeightRatio, pitch, cv=kfold)
# print("Mean %.2f" % results.mean())
