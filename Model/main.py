import os
import sys
from sys import platform


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.layers import LSTM, Dense, Dropout
# caution: path[0] is reserved for script path (or '' in REPL)
# sys.path.insert(1, 'bvh-python-master')

from keras.models import Sequential
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# tf.debugging.set_log_device_placement(True)

script_dir = os.path.abspath('..') #os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'Preprocess', 'BVH')
sys.path.append(mymodule_dir)
from readAll_ACCADbvh import loadBVHdataFrames

from tensorflow.python.keras import backend as K
# adjust values to your needs
config = tf.compat.v1.ConfigProto( device_count = {'GPU': 1 , 'CPU': 8} )
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

def model_Single(lenTs, lenFs):
    model = Sequential()
    model.add(LSTM(30, activation='relu', return_sequences=True, input_shape=(lenTs, lenFs)))
    model.add(Dropout(0.3))
    model.add(LSTM(20, activation='relu', return_sequences=True))
    model.add(LSTM(10, activation='relu'))
    model.add(Dense(10, activation='relu', name='fingerprint'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_Multi(lenTs, lenFs):
    # Define model layers.
    input_layer = keras.Input(shape=(lenTs, lenFs,))
    input_lstm = LSTM(30, activation='relu', return_sequences=True, input_shape=(lenTs, lenFs))(input_layer)
    first_dropout = Dropout(0.3)(input_lstm)
    second_lstm = LSTM(20, activation='relu', return_sequences=True)(first_dropout)
    third_lstm = LSTM(10, activation='relu')(second_lstm)
    pre_fingerprint_dense1 = Dense(10, activation='relu')(third_lstm)
    fingerprint_layer = Dense(64, activation='relu', name='fingerprint')(pre_fingerprint_dense1)
    # fingerprint_dropout = Dropout(0.3)(fingerprint_layer)

    class_dense1 = Dense(32, activation='relu')(fingerprint_layer)
    class_dense2 = Dense(16, activation='relu')(class_dense1)
    class_output = Dense(3, activation='softmax', name='class_output')(class_dense2)

    identity_dense1 = Dense(32,activation='relu')(fingerprint_layer)
    identity_dense2 = Dense(16, activation='relu')(identity_dense1)
    identity_output = Dense(3,activation='softmax', name='identity_output')(identity_dense2)
    # Define the model with the input layer
    # and a list of output layers
    model = keras.Model(inputs=input_layer,outputs=[class_output, identity_output])
    # Specify the optimizer, and compile the model with loss functions for both outputs
    model.compile(optimizer='adam',
                  loss={'class_output': 'categorical_crossentropy', 'identity_output': 'categorical_crossentropy'},
                  metrics={'class_output': 'accuracy',
                           'identity_output': 'accuracy'})
    return model


def model_Multi2(lenTs, lenFs):
    # Define model layers.
    input_layer = keras.Input(shape=(lenTs, lenFs,))
    input_lstm = LSTM(30, activation='relu', return_sequences=True, input_shape=(lenTs, lenFs))(input_layer)
    first_dropout = Dropout(0.3)(input_lstm)
    second_lstm = LSTM(20, activation='relu', return_sequences=True)(first_dropout)
    third_lstm = LSTM(10, activation='relu')(second_lstm)
    pre_fingerprint_dense1 = Dense(10, activation='relu')(third_lstm)
    fingerprint_layer = Dense(64, activation='relu', name='fingerprint')(pre_fingerprint_dense1)
    fingerprint_dropout = Dropout(0.3)(fingerprint_layer)

    class_dense1 = Dense(32, activation='relu')(fingerprint_dropout)
    class_dense2 = Dense(16, activation='relu')(class_dense1)
    class_output = Dense(6, activation='softmax', name='class_output')(class_dense2)

    identity_dense1 = Dense(32,activation='relu')(fingerprint_layer)
    identity_dense2 = Dense(16, activation='relu')(identity_dense1)
    identity_output = Dense(3,activation='softmax', name='identity_output')(identity_dense2)
    # Define the model with the input layer
    # and a list of output layers
    model = keras.Model(inputs=input_layer,outputs=[class_output, identity_output])
    # Specify the optimizer, and compile the model with loss functions for both outputs
    model.compile(optimizer='adam',
                  loss={'class_output': 'categorical_crossentropy', 'identity_output': 'categorical_crossentropy'},
                  metrics={'class_output': 'accuracy',
                           'identity_output': 'accuracy'})
    return model


def model_Multi3(lenTs, lenFs, No_Act_class = 6, No_ID_class = 3):
    # Define model layers.
    input_layer = keras.Input(shape=(lenTs, lenFs,))
    input_lstm = LSTM(30, activation='relu', return_sequences=True, input_shape=(lenTs, lenFs))(input_layer)
    first_dropout = Dropout(0.3)(input_lstm)
    second_lstm = LSTM(20, activation='relu', return_sequences=True)(first_dropout)
    third_lstm = LSTM(10, activation='relu')(second_lstm)
    pre_fingerprint_dense1 = Dense(10, activation='relu')(third_lstm)
    fingerprint_layer = Dense(64, activation='relu', name='fingerprint')(pre_fingerprint_dense1)
    fingerprint_dropout = Dropout(0.3)(fingerprint_layer)

    class_dense1 = Dense(32, activation='relu')(fingerprint_dropout)
    class_dense2 = Dense(16, activation='relu')(class_dense1)
    class_output = Dense(No_Act_class, activation='softmax', name='class_output')(class_dense2)

    identity_dense1 = Dense(32,activation='relu')(fingerprint_layer)
    identity_dense2 = Dense(16, activation='relu')(identity_dense1)
    identity_output = Dense(No_ID_class,activation='softmax', name='identity_output')(identity_dense2)
    # Define the model with the input layer
    # and a list of output layers
    model = keras.Model(inputs=input_layer,outputs=[class_output, identity_output])
    # Specify the optimizer, and compile the model with loss functions for both outputs
    model.compile(optimizer='adam',
                  loss={'class_output': 'categorical_crossentropy', 'identity_output': 'categorical_crossentropy'},
                  metrics={'class_output': 'accuracy',
                           'identity_output': 'accuracy'})
    return model

def plotQuiver():
    pass

def plotConfusionMatrix(y_true, y_pred):
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
    #
    # Print the confusion matrix using Matplotlib
    #
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    datasetName = 'ACCAD'  # 'BerkerlyMHAD'
    datasetName = 'BerkerlyMHAD'
    loadAndTrainModel = False
    if loadAndTrainModel:
        loadModelName = 'model5ACT_PERS2'
    modelName = 'model5ACT_PERS'

    if platform == "win32":
        # Windows...
        if datasetName == 'ACCAD':
            path = "C://Users//gawsa//Documents//GitHub//FamtamAI//Data//ACCAD//"
        elif datasetName == 'BerkerlyMHAD':
            path = "C://Users//gawsa//Documents//GitHub//FamtamAI//Data//BerkerlyMHAD//SkeletalData//"
    else:
        # mac
        if datasetName == 'ACCAD':
            path = "//Users//gowthami//documents//GitHub//FamtamAI//Data//ACCAD//"
        elif datasetName == 'BerkerlyMHAD':
            path = "//Users//gowthami//documents//GitHub//FamtamAI//Data//BerkerlyMHAD//SkeletalData//"

    if datasetName == 'ACCAD':
        path = "C://Users//gawsa//Documents//GitHub//FamtamAI//Data//ACCAD//"
        chosenClass = ['Stand', 'Walk', 'Run']
        chosenIdentity = ['Female1', 'Male1', 'Male2']
        lenFs = 69
        lenTs = 30
        subsamplingFactor = 1
    elif datasetName == 'BerkerlyMHAD':
        path = 'C://Users//gawsa//Documents//GitHub//FamtamAI//Data//BerkerlyMHAD//'
        chosenClass = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11']
        chosenIdentity = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12']
        lenFs = 93
        lenTs = 1000
        subsamplingFactor = 1

    if chosenIdentity == None:
        dataFrames, dataLabels = loadBVHdataFrames(path, chosenClass, lenTs, lenFs, datasetName)
    else:
        dataFrames, dataLabels = loadBVHdataFrames(path, chosenClass, lenTs, lenFs, datasetName, chosenIdentity, subsamplingFactor)

    print('Final Data:', len(dataLabels),',',len(dataLabels[0]) ,',',len(dataLabels[1]) )
    print(dataLabels)
    print(dataFrames)

    # # save to csv file
    # np.savetxt('dataFrames.csv', dataFrames, delimiter=',')
    # np.savetxt('dataLabels.csv', dataLabels, delimiter=',')

    # # load array
    # dataFrames = np.loadtxt('dataFrames.csv', delimiter=',')
    # dataLabels = np.loadtxt('dataLabels.csv', delimiter=',')

    X = dataFrames
    Y = [keras.utils.to_categorical([a[0] for a in dataLabels], num_classes=len(chosenClass)),
            keras.utils.to_categorical([a[1] for a in dataLabels], num_classes=len(chosenIdentity))]
    print(Y)

    if loadAndTrainModel:
        model = keras.models.load_model(loadModelName)
    else:
        if datasetName == 'ACCAD':
            if chosenIdentity == None:
                model = model_Single(lenTs, lenFs)
            else:
                # model = model_Multi(lenTs, lenFs)
                model = model_Multi2(lenTs, lenFs)
        elif datasetName == 'BerkerlyMHAD':
            if chosenIdentity == None:
                print('No model Designed')
            else:
                model = model_Multi3(lenTs, lenFs, len(chosenClass), len(chosenIdentity))


    print(model.summary())

    if chosenIdentity == None:
        model.fit(X, Y, epochs=250, batch_size=4)
        score = model.evaluate(X, Y, batch_size=4)
    else:
        model.fit(X, (Y[0],Y[1]), epochs=500, batch_size=1)
        score = model.evaluate(x=X, y=(Y[0],Y[1]))
    print(score)

    # Calling `save('my_model')` creates a SavedModel folder `my_model`.
    model.save(modelName)

    # It can be used to reconstruct the model identically.
    if chosenIdentity == None:
        reconstructed_model = keras.models.load_model("my_model")
        Y_Score = reconstructed_model.predict(X)
        Y_Label = Y_Score.argmax(axis=-1)
    else:
        reconstructed_model = keras.models.load_model(modelName)
        Y_Score = reconstructed_model.predict(X)
        Y_Label = (Y_Score[0].argmax(axis=-1), Y_Score[1].argmax(axis=-1))

    #
    # Calculate the confusion matrix
    #
    plotConfusionMatrix(y_true=list([a[0] for a in dataLabels]), y_pred=list(Y_Label[0]))


    # print("Y_Label:", Y_Label)

    fingerprint = Sequential()
    for i in reconstructed_model.layers:
        fingerprint.add(i)
        if i.name == "fingerprint":
            break
    fingerprint.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    Y_fingerprint = fingerprint(X)
    # print(Y_fingerprint)

    no_of_plots = 12
    for ifig in range(6):

        # Initialise the subplot function using number of rows and columns
        figure, axis = plt.subplots(int(no_of_plots/3),3)
        figure.tight_layout(pad=1.0)

        selIdx = -1
        for axess in axis:
            for axes in axess:
                selIdx = selIdx + 1

                pltstyle = 3

                fingerprint_X = [0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
                fingerprint_Y = [2, 2, 2, 1, 1, 1, 0, 0, 0, -1]

                if pltstyle == 1:
                    cnt = 0
                    for i in zip(fingerprint_X, fingerprint_Y):
                        axes.quiver(i[0], i[1],
                                2 + Y_fingerprint.numpy()[ifig*no_of_plots + selIdx, cnt],
                                2 + Y_fingerprint.numpy()[ifig*no_of_plots + selIdx, ((cnt + 5) % 10)])
                        cnt = cnt + 1

                elif pltstyle == 2:
                    colors = np.random.rand(10)
                    fingerprintPlot = np.zeros([2,])
                    area = 100 * (Y_fingerprint.numpy()[ifig*no_of_plots + selIdx, 0::1])  # 0 to 15 point radii
                    axes.scatter(fingerprint_X, fingerprint_Y, s=area, alpha=0.5)
                    # axes.plot(Y_fingerprint.numpy()[selIdx, 0::2], Y_fingerprint.numpy()[selIdx, 1::2])
                elif pltstyle == 3:
                    fingerprintPlotX, fingerprintPlotY = [0], [0]
                    xtoggle, ytoggle = True, True
                    cnt = 0
                    x_val, y_val = 0, 0
                    for val in Y_fingerprint.numpy()[ifig*no_of_plots + selIdx, 0::1]:
                        if xtoggle:
                            if ytoggle:
                                y_val = y_val + val
                                fingerprintPlotX.append(-x_val)
                                fingerprintPlotY.append(y_val)
                                ytoggle = False
                            else:
                                x_val = x_val + val
                                fingerprintPlotX.append(x_val)
                                fingerprintPlotY.append(y_val)
                                xtoggle = False
                        else:
                            if ytoggle:
                                x_val = x_val + val
                                fingerprintPlotX.append(-x_val)
                                fingerprintPlotY.append(-y_val)
                                xtoggle = True
                            else:
                                y_val = y_val + val
                                fingerprintPlotX.append(x_val)
                                fingerprintPlotY.append(-y_val)
                                ytoggle = True

                    axes.plot(fingerprintPlotX,fingerprintPlotY )

                else:
                    axes.plot(Y_fingerprint.numpy()[ifig*no_of_plots + selIdx, :])

                if chosenIdentity == None:
                    axes.set_title(chosenClass[Y_Label[selIdx]] + "/" + chosenClass[dataLabels[selIdx]])
                else:
                    axes.set_title(chosenIdentity[Y_Label[1][selIdx]] + "_" + chosenClass[Y_Label[0][selIdx]]
                                   + "/" + chosenIdentity[dataLabels[selIdx][1]] + "_" + chosenClass[dataLabels[selIdx][0]])

        plt.show()


    # Y_predict = reconstructed_model(X)
    # print(Y_predict);

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
