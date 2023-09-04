import math
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize, StandardScaler
from tabulate import tabulate

# caution: path[0] is reserved for script path (or '' in REPL)
script_dir = os.path.dirname(__file__)
mymodule_dir = os.path.join(script_dir, 'bvh-python-master')
sys.path.append(mymodule_dir)
from bvh import Bvh

# read using the libray
def readBVH(doc_name):
    with open(doc_name) as f:
        mocap = Bvh(f.read())
    return mocap

# process the mocap.frames and return as numpy 2D array of numbers
def split_processFrames(mocap):
    # def split the array into positions and rotations
    def splitFrameRotvsPos(Frames, splitIdx):
        PosFrames = Frames[:splitIdx, :]
        RotFrames = Frames[splitIdx:, :]
        return [PosFrames, RotFrames]

    nFrames = len(mocap.frames)
    nSize = len(mocap.frames[0])
    print('Input Frame size:', nFrames, nSize)
    jointNames = mocap.get_joints_names()
    #print('Joint Names:', jointNames)
    channelNames = mocap.get_channel_names()
    #print('Channel Names:', channelNames)
    PosFrame_Title = ["_".join([jointNames[0], a]) for a in channelNames[0][:3]]
    #print('PosFrame_Title:', PosFrame_Title)
    RotFrame_Title = ["_".join([b,a]) for b in jointNames for list in channelNames for a in list[3:]]
    #print('RotFrame_Title:', RotFrame_Title)

    Frames, Frame_Title = processFrames(mocap)

    PosFrames, RotFrames = splitFrameRotvsPos(Frames, 3)   #First 3 are positions

    return PosFrames, RotFrames, PosFrame_Title, RotFrame_Title

def processFrames(mocap):

    nFrames = len(mocap.frames)
    nSize = len(mocap.frames[0])
    # print('Input Frame size:', nFrames, nSize)
    jointNames = mocap.get_joints_names()
    #print('Joint Names:', jointNames)
    channelNames = mocap.get_channel_names()
    #print('Channel Names:', channelNames)
    Frame_Title = ["_".join([jointNames[0], a]) for a in channelNames[0][:3]]
    Frame_Title.extend(["_".join([b,a]) for b in jointNames for list in channelNames for a in list[3:]])
    #print('RotFrame_Title:', Frame_Title)

    Frames = np.zeros((nSize, nFrames))
    i_cols, i_rows = 0,0
    for cols in mocap.frames:
        i_rows = 0
        for rows in cols:
            Frames[i_rows, i_cols] = float(rows)
            i_rows = i_rows+1
        i_cols = i_cols + 1

    return Frames, Frame_Title

def getPandasDF(aFrame, aFrame_Title):
    DF = pd.DataFrame(np.transpose(aFrame))
    DF.columns = aFrame_Title
    # print('DF Any Null: ', DF.isnull().any().sum())
    print(tabulate(DF.head(2), headers='keys'))
    # print(DF.describe())
    return DF

def normalizedPandasDF(df):
    for key, value in df.items():
        df[key] = StandardScaler().fit_transform(df[key].values.reshape(-1, 1))
    print(tabulate(df.head(2), headers='keys'))
    #print(df.describe())
    return df

if __name__ == '__main__':

    path = 'C:/Users/gawsa/Documents/GitHub/FamtamAI/Data/ACCAD/Male1_bvh/Male1_A1_Stand.bvh'
    path = 'C:/Users/gawsa/Documents/GitHub/FamtamAI/Data/BerkerlyMHAD/SkeletalData/skl_s01_a01_r01.bvh'
    # path = 'bvh-python-master/tests/test_mocapbank.bvh'
    mocap = readBVH(path)

    Frames, Frame_Title = processFrames(mocap)
    print('Frames:', Frames)

    # PosFrames, RotFrames, PosFrame_Title, RotFrame_Title = split_processFrames(mocap)
    #print(Frames.shape)
    #print(PosFrames.shape)
    #print(RotFrames.shape)

    # print('Position:')
    # posDF = getPandasDF(PosFrames, PosFrame_Title)
    # posDF = normalizedPandasDF(posDF)
    #
    # print('Rotation:')
    # rotDF = getPandasDF(RotFrames, RotFrame_Title)
    # rotDF = normalizedPandasDF(rotDF)


