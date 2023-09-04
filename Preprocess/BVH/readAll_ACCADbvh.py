# import OS module
import math
import os

import numpy as np

from readBVH import readBVH, processFrames, getPandasDF, normalizedPandasDF

def getAllFilenames(path):
    # Get the list of all files and directories
    dir_list = os.listdir(path)
    return dir_list

def getBVHnumpy(filepath):
    mocap = readBVH(filepath)
    Frames, Frame_Title = processFrames(mocap)
    # print(Frame_Title)
    # print(Frames)
    return Frames, Frame_Title

def getBVHPandasDF(filepath):
    Frames, Frame_Title = getBVHnumpy(filepath)

    DF = getPandasDF(Frames, Frame_Title)
    # DF = normalizedPandasDF(DF)

    return DF

def padNumpyFrame(Frame, lenTs, no_of_channels, subsamplingFactor):

    lenFrame = int(Frame.shape[1]/subsamplingFactor)

    if (lenFrame % lenTs)/lenTs >= 0.8:
        shape = (int(lenFrame/(lenTs)) + 1, lenTs, no_of_channels)
    else:
        shape = (int(lenFrame / (lenTs)), lenTs, no_of_channels)

    retNpArr = np.zeros(shape)

    # print(Frame.shape[1])
    for i in range(shape[0]):
        adjLenFrame = lenFrame * subsamplingFactor
        adjlenTS = lenTs * subsamplingFactor
        if lenFrame <= lenTs:
            retNpArr[i,-lenFrame:,:] = Frame[:,-adjLenFrame::subsamplingFactor,:]
        else:
            retNpArr[i,:,:] = Frame[:,i*adjlenTS:(i+1)*adjlenTS:subsamplingFactor,:]
            lenFrame = lenFrame - lenTs

    return retNpArr

def standardizeFrames(Frames):

    nSample, nFrames, nSize = Frames.shape

    FramesX = np.zeros((nSample, nFrames, nSize))
    for ii_samp in range(nSample):
        for ii_cols in range(nFrames):
            for ii_rows in range(nSize):
                if ii_rows < 3:
                    FramesX[ii_samp, ii_cols, ii_rows] = Frames[ii_samp, ii_cols, ii_rows] - Frames[ii_samp, 0, ii_rows]
                else:
                    FramesX[ii_samp, ii_cols, ii_rows] = (Frames[ii_samp, ii_cols, ii_rows] - Frames[ii_samp, 0, ii_rows]) * math.pi / 180

    FramesX = np.array(FramesX)

    return FramesX

def loadBVHdataFrames(path, chosenClass, lenTs=100, no_of_channels = 93, datasetName = 'BerkerlyMHAD', chosenIdentity = None, subsamplingFactor = 1):
    dataFrames = []
    dataLabels = []
    folder_list = getAllFilenames(path)

    for folder in folder_list:
        folderPath = "".join([path, folder])
        dir_list = getAllFilenames(folderPath)

        print("Files and directories in '", folderPath, "' :")
        # prints all files
        # print(dir_list)

        for i in dir_list:

            if datasetName == 'ACCAD':
                className = i.split('_')[-1].split('.')[0]
                identity = i.split('_')[0]
            elif datasetName == 'BerkerlyMHAD':
                className = i.split('_')[-2].split('.')[0]
                identity = i.split('_')[1]

            if className in chosenClass:
                Label = chosenClass.index(className, 0, len(chosenClass))
                if chosenIdentity != None:
                    Label2 = chosenIdentity.index(identity, 0, len(chosenIdentity))
                # print("Class:", className)
                # print("Label:", Label)

                filePath = "//".join([folderPath, i])
                # print('filePath:',filePath)

                Frames, Frame_Title = getBVHnumpy(filePath)
                Frames = np.transpose(Frames).reshape((1,-1,no_of_channels))

                paddedFrames = padNumpyFrame(Frames, lenTs, no_of_channels, subsamplingFactor)
                paddedFrames = standardizeFrames(paddedFrames)
                # print(paddedFrames.shape)
                # print(paddedFrames)
                # print()

                if len(dataFrames) == 0:
                    dataFrames = paddedFrames
                    if chosenIdentity == None:
                        dataLabels = [Label]
                    else:
                        dataLabels = [[Label, Label2]]
                    for i in range(paddedFrames.shape[0] - 1):
                        if chosenIdentity == None:
                            dataLabels.append(Label)
                        else:
                            dataLabels.append([Label, Label2])
                else:
                    dataFrames = np.append(dataFrames, paddedFrames, axis=0)
                    if chosenIdentity == None:
                        dataLabels.append(Label)
                    else:
                        dataLabels.append([Label, Label2])
                    for i in range(paddedFrames.shape[0] - 1):
                        if chosenIdentity == None:
                            dataLabels.append(Label)
                        else:
                            dataLabels.append([Label, Label2])

    return dataFrames, dataLabels

if __name__ == '__main__':

    dataset = 'ACCAD' # 'BerkerlyMHAD'
    dataset = 'BerkerlyMHAD'

    if dataset == 'ACCAD':
        path = "C://Users//gawsa//Documents//GitHub//FamtamAI//Data//ACCAD//"
        chosenClass = ['Stand', 'Walk', 'Run']
        chosenIdentity = ['Female1', 'Male1', 'Male2']
        no_of_channels = 69
        lenTs = 30
    elif dataset == 'BerkerlyMHAD':
        path = 'C://Users//gawsa//Documents//GitHub//FamtamAI//Data//BerkerlyMHAD//'
        chosenClass = ['a01', 'a02', 'a03', 'a04', 'a05', 'a06', 'a07', 'a08', 'a09', 'a10', 'a11']
        chosenIdentity = ['s01', 's02', 's03', 's04', 's05', 's06', 's07', 's08', 's09', 's10', 's11', 's12']
        no_of_channels = 93
        lenTs = 30

    folder_list = getAllFilenames(path)

    dataFrames = []
    dataLabels = []

    for folder in folder_list:
        folderPath = "".join([path, folder])
        dir_list = getAllFilenames(folderPath)

        print("Files and directories in '", folderPath, "' :")
        # prints all files
        print(dir_list)

        for i in dir_list:

            if dataset == 'ACCAD':
                className = i.split('_')[-1].split('.')[0]
                identity = i.split('_')[0]
            elif dataset == 'BerkerlyMHAD':
                className = i.split('_')[-2].split('.')[0]
                identity = i.split('_')[1]

            if className in chosenClass:
                Label = chosenClass.index(className, 0, len(chosenClass))
                Label2 = chosenIdentity.index(identity, 0, len(chosenIdentity))
                print("Class:", className)
                print("Label:", Label, ', ', Label2)

                filePath = "//".join([folderPath, i])
                # print('filePath:',filePath)

                Frames, Frame_Title = getBVHnumpy(filePath)

                Frames = np.transpose(Frames).reshape((1,-1,no_of_channels))

                print(Frames.shape)
                print(Frames)
                # npData = getBVHPandasDF(filePath)
                # classDF

                paddedFrames = padNumpyFrame(Frames, lenTs, no_of_channels)
                paddedFrames = standardizeFrames(paddedFrames)
                # print(paddedFrames.shape)
                print(paddedFrames)

                print()
                if len(dataFrames) == 0:
                    dataFrames = paddedFrames
                    dataLabels = [[Label, Label2]]
                    for i in range(paddedFrames.shape[0] - 1):
                        dataLabels.append([Label,Label2])
                else:
                    dataFrames = np.append(dataFrames, paddedFrames, axis=0)
                    dataLabels.append([Label,Label2])
                    for i in range(paddedFrames.shape[0] - 1):
                        dataLabels.append([Label,Label2])

                # break
        print("Final Dataset:", dataLabels)
        print(dataFrames.shape)