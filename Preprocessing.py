import sys
import pickle
from numpy import *
import numpy as np
import matplotlib.pyplot as plt
import collections 
from scipy import signal
import pandas as pd 

# sample rate
numpacket = 125

policy = ['gx', 'gy', 'gz', 'ax', 'ay', 'az', 'rgax', "rgay", "rgaz"]
nodes = ['wrist', 'waist', 'ankle']

subjectList = ["Wiz", "Roger", "James", "Ting", "Ryan", "Hank"]

activityList = [
    "Sit", 
    "Stand", 
    "Walk", 
    "Upstairs", 
    "Downstairs", 
    "Drink_Water", 
    "Brush_Teeth", 
    "Cleaner", 
    "Jogging", 
    "Open_Door", 
    "Stretch", 
    "Lie", 
    "Walk_Phone"
]

lengthList = [15, 15, 15, 15, 15, 15]

def read_pickle(index, pathHead, sec):

    fp1 = open(pathHead + str(index) + "wrist.dat", "rb")
    fp2 = open(pathHead + str(index) + "waist.dat", "rb")
    fp3 = open(pathHead + str(index) + "ankle.dat", "rb")

    objs1 = []
    objs2 = []
    objs3 = []
    
    while True:
        try:
            objs1.append(pickle.load(fp1))
        except:
            break

    while True:
        try:
            objs2.append(pickle.load(fp2))
        except:
            break

    while True:
        try:
            objs3.append(pickle.load(fp3))
        except:
            break

    timestamp = list()
    timestampChip = []
    yaw1 = list()
    pitch1 = list()
    roll1 = list()
    gx = list()
    gy = list()
    gz = list()
    ax =list()
    ay =list()
    az =list()
    rgax = list()
    rgay = list()
    rgaz = list()

    timestamp2 = list()
    timestampChip2 = []
    yaw2 = list()
    pitch2 = list()
    roll2 = list()
    gx2 = list()
    gy2 = list()
    gz2 = list()
    ax2 =list()
    ay2 =list()
    az2 =list()
    rgax2 = list()
    rgay2 = list()
    rgaz2 = list()


    timestamp3 = list()
    timestampChip3 = []
    yaw3 = list()
    pitch3 = list()
    roll3 = list()
    gx3 = list()
    gy3 = list()
    gz3 = list()
    ax3 =list()
    ay3 =list()
    az3 =list()
    rgax3 = list()
    rgay3 = list()
    rgaz3 = list()

    for obj in objs1:
        timestamp.append(obj['timestamp'])
        timestampChip.append(obj['timestampChip'])
        yaw1.append(obj['Angle'].item(0,0))
        pitch1.append(obj['Angle'].item(0,1))
        roll1.append(obj['Angle'].item(0,2))
        gz.append(obj['Gyo'].item(0, 2))
        gy.append(obj['Gyo'].item(0, 1))
        gx.append(obj['Gyo'].item(0, 0))
        ax.append(obj['Acc'].item(0, 0))
        ay.append(obj['Acc'].item(0, 1))
        az.append(obj['Acc'].item(0, 2))
        rgax.append(obj['rgAcc'].item(0, 0))
        rgay.append(obj['rgAcc'].item(0, 1))
        rgaz.append(obj['rgAcc'].item(0, 2))

    for obj in objs2:
        timestamp2.append(obj['timestamp'])
        timestampChip2.append(obj['timestampChip'])
        yaw2.append(obj['Angle'].item(0,0))
        pitch2.append(obj['Angle'].item(0,1))
        roll2.append(obj['Angle'].item(0,2))
        gz2.append(obj['Gyo'].item(0, 2))
        gy2.append(obj['Gyo'].item(0, 1))
        gx2.append(obj['Gyo'].item(0, 0))
        ax2.append(obj['Acc'].item(0, 0))
        ay2.append(obj['Acc'].item(0, 1))
        az2.append(obj['Acc'].item(0, 2))
        rgax2.append(obj['rgAcc'].item(0, 0))
        rgay2.append(obj['rgAcc'].item(0, 1))
        rgaz2.append(obj['rgAcc'].item(0, 2))

    for obj in objs3:
        timestamp3.append(obj['timestamp'])
        timestampChip3.append(obj['timestampChip'])
        yaw3.append(obj['Angle'].item(0,0))
        pitch3.append(obj['Angle'].item(0,1))
        roll3.append(obj['Angle'].item(0,2))
        gz3.append(obj['Gyo'].item(0, 2))
        gy3.append(obj['Gyo'].item(0, 1))
        gx3.append(obj['Gyo'].item(0, 0))
        ax3.append(obj['Acc'].item(0, 0))
        ay3.append(obj['Acc'].item(0, 1))
        az3.append(obj['Acc'].item(0, 2))
        rgax3.append(obj['rgAcc'].item(0, 0))
        rgay3.append(obj['rgAcc'].item(0, 1))
        rgaz3.append(obj['rgAcc'].item(0, 2))

    totalpacket = sec * numpacket

    timestamp = np.linspace(0, sec, totalpacket)
    timestamp2 = np.linspace(0, sec, totalpacket)
    timestamp3 = np.linspace(0, sec, totalpacket)
    
    gx = length_normalize(gx, totalpacket)
    gy = length_normalize(gy, totalpacket)
    gz = length_normalize(gz, totalpacket)
    ax = length_normalize(ax, totalpacket)
    ay = length_normalize(ay, totalpacket)
    az = length_normalize(az, totalpacket)
    rgax = length_normalize(rgax, totalpacket)
    rgay = length_normalize(rgay, totalpacket)
    rgaz = length_normalize(rgaz, totalpacket)

    gx2 = length_normalize(gx2, totalpacket)
    gy2 = length_normalize(gy2, totalpacket)
    gz2 = length_normalize(gz2, totalpacket)
    ax2 = length_normalize(ax2, totalpacket)
    ay2 = length_normalize(ay2, totalpacket)
    az2 = length_normalize(az2, totalpacket)
    rgax2 = length_normalize(rgax2, totalpacket)
    rgay2 = length_normalize(rgay2, totalpacket)
    rgaz2 = length_normalize(rgaz2, totalpacket)

    gx3 = length_normalize(gx3, totalpacket)
    gy3 = length_normalize(gy3, totalpacket)
    gz3 = length_normalize(gz3, totalpacket)
    ax3 = length_normalize(ax3, totalpacket)
    ay3 = length_normalize(ay3, totalpacket)
    az3 = length_normalize(az3, totalpacket)
    rgax3 = length_normalize(rgax3, totalpacket)
    rgay3 = length_normalize(rgay3, totalpacket)
    rgaz3 = length_normalize(rgaz3, totalpacket)

    gx, gy, gz, ax, ay, az, rgax, rgay, rgaz = low_pass([gx, gy, gz, ax, ay, az, rgax, rgay, rgaz])
    gx2, gy2, gz2, ax2, ay2, az2, rgax2, rgay2, rgaz2 = low_pass([gx2, gy2, gz2, ax2, ay2, az2, rgax2, rgay2, rgaz2])
    gx3, gy3, gz3, ax3, ay3, az3, rgax3, rgay3, rgaz3 = low_pass([gx3, gy3, gz3, ax3, ay3, az3, rgax3, rgay3, rgaz3])
    
    sensorData = [timestamp, gx, gy, gz, ax, ay, az, rgax, rgay, rgaz]
    sensorData2 = [timestamp2, gx2, gy2, gz2, ax2, ay2, az2, rgax2, rgay2, rgaz2]
    sensorData3 = [timestamp3, gx3, gy3, gz3, ax3, ay3, az3, rgax3, rgay3, rgaz3]
    
    return [sensorData, sensorData2, sensorData3]

def low_pass(dataRaw):
    #     nyq = 0.5 * 125
    #     normal_cutoff = 2.2 / nyq
    #     b, a = signal.butter(3, normal_cutoff, btype = 'low', analog = False)
    # 40Hz=>1, 50Hz=>0.8, 75Hz=>0.53, 100Hz=>0.4, 125Hz=>0.3 

    b, a = signal.butter(10, 0.3)
    returnList = []
    for dataList in dataRaw:
        dataFilt = signal.filtfilt(b, a, dataList)
        returnList.append(dataFilt[:])
    return returnList

# map acc and gyro to [0,1]
def normalize(dest, axis, max_value, min_value):
    for i, _ in enumerate(policy):
        if _ == axis:
            MAX = max_value[i]
            MIN = min_value[i]

    for i, v in enumerate(dest):
        dest[i] = (v-MIN)/(MAX-MIN)
    return dest

# our system has 3 nodes, so we need to resample numbers of sample to same value
def length_normalize(src, nor_len=200):
    # linear interpolation 
    dest = [0 for i in range(nor_len)]
    org_len = len(src)
    for i in range(nor_len):
        expected_index = float(i)*org_len/nor_len
        real_index = i*org_len/nor_len
        dist = expected_index-real_index
        if real_index < len(src)-1:
            dest[i] = src[int(real_index)+1]*(dist) + src[int(real_index)]*(1-dist)
        else:
            dest[i] = src[int(real_index)]
    return dest

# sliding window
def windows(data, size):
    start = 0
    while start < len(data):
        yield int(start), int(start + size)
        # Overlapping 50%
        start += (0.5*size) 

def segment_signal(data, window_size = 400, index = 1):
    segments = []
    data_timestamp = data[0]
    for (start, end) in windows(data_timestamp, window_size):
        tmp = data[index][start:end]
        if(len(data_timestamp[start:end]) == window_size):
            segments.append(tmp)
    return segments

# for normalization, we must find the maximum and minimum 
def find_maxmin(dataList):
    maxList = {
        "wrist":{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        },
        'waist':{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        }, 
        'ankle':{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        }
    }

    minList = {
        "wrist":{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[], 
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        },
        'waist':{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        }, 
        'ankle':{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        }
    }

    for i, name in enumerate(policy): 
        tmp = dataList.loc[:,name]
        for j in range(len(tmp)):
            if j % 3 == 0:
                maxList['wrist'][name].append(max(tmp[j]))
                minList['wrist'][name].append(min(tmp[j]))
            elif j % 3 == 1:
                maxList['waist'][name].append(max(tmp[j]))
                minList['waist'][name].append(min(tmp[j]))
            else:
                maxList['ankle'][name].append(max(tmp[j]))
                minList['ankle'][name].append(min(tmp[j]))

    max_value = {'wrist':[], 'waist':[], 'ankle':[]}
    min_value = {'wrist':[], 'waist':[], 'ankle':[]}
    
    for m, node in enumerate(nodes):
        for n, ttype in enumerate(policy):
        	max_value[node].append(max(maxList[node][ttype]))
        	min_value[node].append(min(minList[node][ttype]))
    return max_value, min_value

def write_to_txt(trainData, fileName, part, axis):
    save_path = 'Data'  + '/'+ fileName + '.txt'
    np.savetxt(save_path,trainData[part][axis])
    print("finish saving file: %s" % fileName)

if __name__ == '__main__':

    dataList = collections.OrderedDict()

    print('read data...')
    test__ = []
    df = []
    policys = ['timestamp', 'gx', 'gy', 'gz', 'ax', 'ay', 'az', 'rgax', "rgay", "rgaz"]
    for i, name in enumerate(subjectList):
        dataLen = lengthList[i]
        for j, activity in enumerate(activityList):
            savedpath = 'Data' + '/' + name + '/' + activity + '/'
            if activity in ["Walk", "Jogging", "Upstairs", "Downstairs", "Brush_Teeth", "Cleaner", "Lie", "Walk_Phone"]: 
                totalsec = 6
            elif activity in ["Sit", "Stand", "Drink_Water", "Open_Door", "Stretch"]: 
                totalsec = 4
            totalpacket = totalsec * numpacket
            for k in range(1, dataLen+1):
                dataList[k-1] = read_pickle(index = k , pathHead = savedpath, sec = totalsec)
                for m, node in enumerate(nodes):
            	    test__.append(name +'_'+ activity + '_'+ str(k) + node)
                df.append(pd.DataFrame(dataList[k-1], index = test__, columns = policys))
                test__ = []

    res = pd.concat(df, axis=0)
    print("end")
    #-------------- Find Max and Min value --------------
    # max_value = ['wrist:gx~rgaz', 'waist:gx~rgaz', 'ankle:gx~rgaz'] total len of value = 27 (3*9)
    print('find maximum and minimum value')
    max_value, min_value= find_maxmin(res)
   
    # -------------- Normalize except timestamp, so i need to add 1--------------
    print('normalize except timestamp')
    for i, axis in enumerate(policy): 
        tmp = res.loc[:,axis]
        for j in range(len(tmp)):
            if j % 3 == 0: # wrist 0, 3, 6....
                res.iloc[j, i+1] = normalize(res.iloc[j, i+1], axis, max_value['wrist'], min_value['wrist'])
            elif j % 3 == 1 : # waist 1, 4, 7....
                res.iloc[j, i+1] = normalize(res.iloc[j, i+1], axis, max_value['waist'], min_value['waist'])
            else: # ankel 2, 5, 8....
                res.iloc[j, i+1] = normalize(res.iloc[j, i+1], axis, max_value['ankle'], min_value['ankle'])

    # -------------- Segementation --------------
    print('segementation')
    windsize = 313 # (2.5s)
    res_window = res.copy()
    rows = res_window.iloc[:, 0].size
    for i in range(rows):
        for j, axis in enumerate(policy):
            res_window.iloc[i, j+1] = segment_signal(res_window.iloc[i, :], window_size = windsize, index = j+1)
    
    # save sample data
    trainData = {
        "wrist":{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        },
        'waist':{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        }, 
        'ankle':{"gx":[],
                 "gy":[],
                 "gz":[],
                 "ax":[],
                 "ay":[],
                 "az":[],
                 "rgax":[],
                 "rgay":[],
                 "rgaz":[]
        }
    }

    for i, axis in enumerate(policy): 
        for j in range(rows):
            if j % 3 == 0: # wrist 0, 3, 6....
                for k in range(len(res_window.iloc[j, i+1])):
                    trainData['wrist'][axis].append(res_window.iloc[j, i+1][k])
            elif j % 3 == 1 : # waist 1, 4, 7....
                for k in range(len(res_window.iloc[j, i+1])):
                    trainData['waist'][axis].append(res_window.iloc[j, i+1][k])
            else: # ankel 2, 5, 8....
                for k in range(len(res_window.iloc[j, i+1])):
                    trainData['ankle'][axis].append(res_window.iloc[j, i+1][k])

    txtName = [ 
                'body_wrist_gyro_x', 'body_wrist_gyro_y', 'body_wrist_gyro_z',
                'body_wrist_acc_x', 'body_wrist_acc_y', 'body_wrist_acc_z',
                'body_wrist_rgacc_x', 'body_wrist_rgacc_y', 'body_wrist_rgacc_z', 
                'body_waist_gyro_x', 'body_waist_gyro_y', 'body_waist_gyro_z',
                'body_waist_acc_x', 'body_waist_acc_y', 'body_waist_acc_z',
                'body_waist_rgacc_x', 'body_waist_rgacc_y', 'body_waist_rgacc_z',
                'body_ankle_gyro_x', 'body_ankle_gyro_y', 'body_ankle_gyro_z',
                'body_ankle_acc_x', 'body_ankle_acc_y', 'body_ankle_acc_z', 
                'body_ankle_rgacc_x', 'body_ankle_rgacc_y', 'body_ankle_rgacc_z'
              ]

    count = 0
    for i, node in enumerate(nodes):
        for j, axis in enumerate(policy): 
            print(node, axis)
            write_to_txt(trainData, txtName[count], node, axis)
            count += 1

    # save data of label
    segment_len = list()
    for i, activity in enumerate(activityList):
        tmp = res_window.loc["Wiz_" + activity +"_1wrist", "gx"] # numbers of windows
        tmp_len = len(tmp) * 15 # numbers of windows * 15 times
        segment_len.append(tmp_len)

    tmp_list = list()
    for i, j in zip(segment_len, [activity+1 for activity in range(len(activityList))]):
        tmp_list.append(np.full((i), j))

    # 13 activity label for 1 person
    y_label = np.hstack(tmp_list[i] for i in range(len(tmp_list)))

    # totally 6 people
    y_label = np.hstack(y_label for i in range(len(subjectList)))

    np.savetxt('Data' + '/' + "y" + '.txt',y_label, fmt="%d")
    print("finish saving file")






