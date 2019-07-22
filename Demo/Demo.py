from bluepy import btle
import struct
import threading
import binascii
import time
import sys
from numpy import *
import numpy as np 
import pickle
import select
import collections 
from scipy import signal
import tensorflow as tf

imuNodeNum = 3
ifaceOffset = 1
dongleNum = 3

acc_divider = 4095.999718
gyro_divider = 65.500002
DEG2RAD = 0.01745329251

lock = threading.Lock()
mainLock = threading.Lock()

# remove gravity parameter
alpha = 0.8

save_data_flag="notReady"
savedFileNum = 0

deviceList = []
dataList = [[], [], []]
window_size = 313
classify_interval = 156

secActivity = 20 
maxList = [[8.080922946418836, 7.50027048537256, 7.193819945725072, 2.172817253154938, 1.9081491682544107, 1.825640798832055, 1.6213674676648793, 1.3715200130109777, 1.0647660595485087], [3.104107950711068, 9.56759082035754, 3.874452965885659, 6.6827239654262485, 1.9755989713420476, 3.7964862109903845, 3.995068764810983, 1.1529348903292325, 1.9654085645288426], [10.995769431735448, 11.087014672734375, 9.65802767226748, 6.561361345980365, 4.912197470231877, 5.721497304337974, 3.908690952491977, 5.09553933600694, 4.563687449936138]]
minList = [[-8.949635810781512, -8.38344766694806, -8.794761715610079, -4.431475455598586, -5.086135937945605, -3.0590418619374433, -2.079319053748967, -2.443361123366064, -1.487472507434863], [-2.770988243920212, -8.319652002350882, -4.299692140307208, -1.9931505032430505, -1.2325619870134386, -1.6323617099013512, -2.077423521830785, -1.0824218126885086, -1.631810126084131], [-8.734417417864824, -12.108411740597216, -7.81672661212909, -3.960178135766902, -7.2771275050784885, -4.541781033824286, -4.415503211585107, -5.433518044681174, -3.484709311784696]]
personStatus = ["Sit", "Standing"]
tmp = "Standing"
dynamicMotion = [
    "Walking", 
    "Upstairs", 
    "Downstairs", 
    "Cleaner", 
    "Jogging", 
    "Walking_Phone",
    "Lie"
]

LABELS = [
    "Sit",
    "Standing", 
    "Walking", 
    "Upstairs", 
    "Downstairs", 
    "Drink_Water",
    "Brush_Teeth",
    "Cleaner", 
    "Jogging", 
    "Open_Door", 
    "Stretch", 
    "Lie",
    "Walking_Phone"
] 

# node's parameter
class myNode(object):
    '''' a class to maintain connection and the calibration value of sensors'''
    def __init__(self):
        self.noti = None
        self.Peripheral = None
        self.accBias = [0.0,0.0,0.0]
        self.gyroBias = [0.0,0.0,0.0]
        self.magBias = [0.0,0.0,0.0]
        self.magScale = [0.0,0.0,0.0]
        self.magCalibration = [0.0,0.0,0.0]
        self.gravity = [0.0,0.0,0.0]
        self.accXList = []
        self.accYList = []
        self.accZList = []
        self.rgaccXList = []
        self.rgaccYList = []
        self.rgaccZList = []
        self.gyroXList = []
        self.gyroYList = []
        self.gyroZList = []
        self.dataDicts= []
        self.lastTime = -1
        # for count packet number
        self.count = 0
 

#get rawdata from ecomini when notification
class MyDelegate(btle.DefaultDelegate):
    def __init__(self, node):
        btle.DefaultDelegate.__init__(self)
        self.node = node

    def handleNotification(self, cHandle, data):
        reading =  binascii.b2a_hex(data) # transfer string to binary then using hex format to express
        self.node.noti = Uint4Toshort([reading[0:4],reading[4:8],reading[8:12],reading[12:16],reading[16:20],reading[20:24],reading[24:28],reading[28:32],reading[32:36],reading[36:40]])
        #self.node.noti means rawdata without calibration from ecomini 
        #acc[x], acc[y], acc[z], gyro[x], gyro[y], gyro[z], mag[x], mag[y], mag[z], timer

#scan
def scanThread():
    global con_start_time
    scanner = btle.Scanner(ifaceOffset)
    iface = 0
    while True :
        if len(deviceList) >= imuNodeNum:
            print("Devices all connected")
            return

        print("Still scanning...")
        try:
            devcies = scanner.scan(timeout = 3)

            # wrist: 3c:cd:40:18:c3:46 (1)
            # waist: 3c:cd:40:18:c0:4f (2)
            # ankle: 3c:cd:40:18:c1:8e (3) 

            for dev in devcies:
                if dev.addr in ["3c:cd:40:18:c3:46", "3c:cd:40:18:c0:4f", "3c:cd:40:18:c1:8e"]: 
               
                    if dev.addr == "3c:cd:40:18:c1:8e" and len(deviceList) != 2:
                        continue
                    elif dev.addr == "3c:cd:40:18:c0:4f" and len(deviceList) != 1:
                        continue

                    print("devcies %s (%s) , RSSI = %d dB" % (dev.addr , dev.addrType , dev.rssi))

                    try:
                        iface = (iface+1)%dongleNum
                        #print("iface = %d" % (iface + ifaceOffset))
                        connNode = myNode()
                        connNode.Peripheral = btle.Peripheral(dev.addr , dev.addrType , iface + ifaceOffset)
                        connNode.Peripheral.setDelegate(MyDelegate(connNode))
                        accBias = binascii.b2a_hex(connNode.Peripheral.readCharacteristic(0x40))
                        gyroBias = binascii.b2a_hex(connNode.Peripheral.readCharacteristic(0x43))
                        magBias = binascii.b2a_hex(connNode.Peripheral.readCharacteristic(0x46))
                        magScale = binascii.b2a_hex(connNode.Peripheral.readCharacteristic(0x49))
                        magCalibration = binascii.b2a_hex(connNode.Peripheral.readCharacteristic(0x4C))
                        calibrationData = [accBias[0:8], accBias[8:16], accBias[16:24]]
                        connNode.accBias = Uint8Tofloat(calibrationData)
                        calibrationData = [gyroBias[0:8], gyroBias[8:16], gyroBias[16:24]]
                        connNode.gyroBias = Uint8Tofloat(calibrationData)
                        calibrationData = [magBias[0:8], magBias[8:16], magBias[16:24]]
                        connNode.magBias = Uint8Tofloat(calibrationData)
                        calibrationData = [magScale[0:8], magScale[8:16], magScale[16:24]]
                        connNode.magScale = Uint8Tofloat(calibrationData)
                        calibrationData = [magCalibration[0:8], magCalibration[8:16], magCalibration[16:24]]
                        connNode.magCalibration = Uint8Tofloat(calibrationData)

                    except:
                        iface = (iface-1)%dongleNum
                        print("failed connection")
                        break

                    #Try to get Service, Characteristic and set notification
                    try:
                        # need to add 0000fed0-0000-1000-8000-00805f9b34fb
                        service = connNode.Peripheral.getServiceByUUID("0000FED0-0000-1000-8000-00805f9b34fb")
                        char = service.getCharacteristics("0000FED7-0000-1000-8000-00805f9b34fb")[0] 
                        connNode.Peripheral.writeCharacteristic(char.handle + 2,struct.pack('<bb', 0x01, 0x00),True)
                        mainLock.acquire()
                        deviceList.append(connNode)
                        mainLock.release()
                    except:
                        print("get service, characteristic or set notification failed")
                        break
        except:
            print "failed scan" 
            exit() 

def detectInputKey():
    global save_data_flag, tStart
    while True:
        if save_data_flag == "notReady":
            print "waiting for 'S' to start collect data"
        i,o,e = select.select([sys.stdin], [], [], 60)
        for s in i:
            if s == sys.stdin:
                input = sys.stdin.readline()
                if input=="S\n":
                    print "start to collect data after 3 seconds"
                    time.sleep(3)
                    print "-----------------------------now start collect data"
                    print "-----------------------------now start collect data"
                    print "-----------------------------now start collect data"
                    save_data_flag = "ready"
                    tStart = time.time()
                    time.sleep(secActivity)
                    print "now end collect data-----------------------------"
                    print "now end collect data-----------------------------"
                    print "now end collect data-----------------------------"
                    #for i, node in enumerate(deviceList):
                    #    print(node.accBias, "count: ", i, node.count)
                    save_data_flag="finish"

def eliminate_gravity(node, value):

    node.gravity[0] = alpha*node.gravity[0] + (1-alpha)*value[0]
    node.gravity[1] = alpha*node.gravity[1] + (1-alpha)*value[1]
    node.gravity[2] = alpha*node.gravity[2] + (1-alpha)*value[2]

    return value[0]-node.gravity[0], value[1]-node.gravity[1], value[2]-node.gravity[2]

def length_normalize(src, nor_len=200):

    # length normalization
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

def imuThread(node, i):
    global save_data_flag, savedFileNum, lock
    print("imuThread start")

    global window_size, dataList, classify_interval, tmp

    file_num = 1
    savedFlag = False

    while True:
        # try:
        if node.Peripheral.waitForNotifications(0.05):
            rawdata = node.noti
            calibratedData = dataCalibration(rawdata, i)
            axrgData, ayrgData, azrgData = eliminate_gravity(node, calibratedData[0:3])
            if save_data_flag == "notReady":
                notReadyFlag = True
                savedFlag = False
            if save_data_flag == "ready":
                if notReadyFlag == True:
                    notReadyFlag = False
                    node.count = 0
                node.accXList.append(calibratedData[0])
                node.accYList.append(calibratedData[1])
                node.accZList.append(calibratedData[2])
                node.rgaccXList.append(axrgData)
                node.rgaccYList.append(ayrgData)
                node.rgaccZList.append(azrgData)
                node.gyroXList.append(calibratedData[3])
                node.gyroYList.append(calibratedData[4])
                node.gyroZList.append(calibratedData[5])
                node.count = node.count + 1
                if len(node.accXList) == window_size:
                    pre_start_time = time.time()
                    accXList, accYList, accZList, rgaccXList, rgaccYList, rgaccZList, gyroXList, gyroYList, gyroZList = lowPass([node.accXList, node.accYList, node.accZList, node.rgaccXList, node.rgaccYList, node.rgaccZList, node.gyroXList, node.gyroYList, node.gyroZList])
                    dataList[i].append(mat([(accXList-minList[i][3])/(maxList[i][3]-minList[i][3]), (accYList-minList[i][4])/(maxList[i][4]-minList[i][4]), (accZList-minList[i][5])/(maxList[i][5]-minList[i][5]), (rgaccXList-minList[i][6])/(maxList[i][6]-minList[i][6]), (rgaccYList-minList[i][7])/(maxList[i][7]-minList[i][7]), (rgaccZList-minList[i][8])/(maxList[i][8]-minList[i][8]), (gyroXList-minList[i][0])/(maxList[i][0]-minList[i][0]), (gyroYList-minList[i][1])/(maxList[i][1]-minList[i][1]), (gyroZList-minList[i][2])/(maxList[i][2]-minList[i][2])]))
                    if len(dataList[0]) > 0 and len(dataList[1]) > 0 and len(dataList[2]) > 0:    
                        predictTh =  threading.Thread(target = predictThread)
                        predictTh.setDaemon(True)
                        predictTh.start()
                    del node.accXList[:classify_interval]                    
                    del node.accYList[:classify_interval]
                    del node.accZList[:classify_interval]
                    del node.rgaccXList[:classify_interval]
                    del node.rgaccYList[:classify_interval]
                    del node.rgaccZList[:classify_interval]
                    del node.gyroXList[:classify_interval]
                    del node.gyroYList[:classify_interval]
                    del node.gyroZList[:classify_interval]
        else:
            continue
        
        if save_data_flag == "finish" and savedFlag == False:
            savedFlag = True
            node.accXList = []
            node.accYList = []
            node.accZList = []
            node.rgaccXList = []
            node.rgaccYList = []
            node.rgaccZList = []
            node.gyroXList = []
            node.gyroYList = []
            node.gyroZList = []
            dataList[i] = []
            lock.acquire()
            savedFileNum = savedFileNum + 1
            if savedFileNum == imuNodeNum:
                save_data_flag = "notReady"
                savedFileNum = 0
                tmp = "Standing"
                print("save_data_flag:", save_data_flag)
            lock.release()
            

def dataCalibration(rawdata, i):
    acc = [None]*3
    acc[0] = rawdata[0]/acc_divider - deviceList[i].accBias[0]
    acc[1] = rawdata[1]/acc_divider - deviceList[i].accBias[1]
    acc[2] = rawdata[2]/acc_divider - deviceList[i].accBias[2]

    gyro = [None]*3
    gyro[0] = (rawdata[3]/gyro_divider - deviceList[i].gyroBias[0])*DEG2RAD
    gyro[1] = (rawdata[4]/gyro_divider - deviceList[i].gyroBias[1])*DEG2RAD
    gyro[2] = (rawdata[5]/gyro_divider - deviceList[i].gyroBias[2])*DEG2RAD

    # secondData = rawdata[7] & (~0xC000)
    secondData = rawdata[9] / 1000000.0

    return [acc[0],acc[1],acc[2],gyro[0],gyro[1],gyro[2],secondData]

def Uint4Toshort(tenData):
    #print(threeData)
    retVal =[]
    
    for i, data in enumerate(tenData):
    #(data)
        i = 0
        byteArray = []
        while(i != 4):
            byteArray.append(int(data[i:i+2], 16))
        #print(int(data, 16))
            i=i+2

        b = ''.join(chr(i) for i in byteArray)
        if i == 9:
            retVal.append(struct.unpack('<H',b)[0])
        else:
            retVal.append(struct.unpack('<h',b)[0])
    return retVal

def Uint8Tofloat(threeData):
    #print(threeData)
    retVal =[]
    
    for data in threeData:
    #(data)
        i = 0
        byteArray = []
        while(i != 8):
            byteArray.append(int(data[i:i+2], 16))
        #print(int(data, 16))
            i=i+2

        b = ''.join(chr(i) for i in byteArray)
        retVal.append(struct.unpack('<f',b)[0])
    return retVal

def lowPass(dataRaw):
    b, a = signal.butter(10, 0.3)
    returnList = []
    for dataList in dataRaw:
        dataFilt = signal.filtfilt(b, a, dataList)
        returnList.append(dataFilt[:])
        
    return returnList

def predictThread():
    global dataList, tflite_model, personStatus, tmp, dynamicMotion

    start_time = time.time()
    model_interpreter_time = 0
    
    input_details = tflite_model.get_input_details()
    output_details = tflite_model.get_output_details()
    input_shape = input_details[0]['shape']
    
    wrist = dataList[0].pop(0)
    waist = dataList[1].pop(0)
    ankle = dataList[2].pop(0)

    data = np.concatenate((wrist.transpose(), waist.transpose(), ankle.transpose()), axis=1)

    num_inputs = data.shape[1]  # 27 input parameters per timestep, height
    num_steps = data.shape[0]      # 313 timesteps per series, width
    data = np.asarray(data)
    # reshape
    X = (data.reshape(1, 1, num_steps, num_inputs)).astype(np.float32)
    X = X.transpose((0, 3, 2, 1))
    # feed one by one
    input_data = X
    # feed data
    tflite_model.set_tensor(input_details[0]['index'], input_data)
    tflite_model.invoke()

    # predict 
    output_data = tflite_model.get_tensor(output_details[0]['index'])

    if np.max(output_data) > 0.9:
        pred_index = np.argmax(output_data, 1)[0]
        motion = LABELS[pred_index]
        if motion in dynamicMotion:
            print("Current_states:{}".format(LABELS[pred_index]))
        else:
            if motion in personStatus:
                tmp = motion
            print("Current_states:{} and {}".format(tmp, LABELS[pred_index]))

def main():
    global runningThreadNum, tflite_model, con_start_time
    #con_start_time = time.time()
    scanTh =  threading.Thread(target = scanThread)
    scanTh.start()
    
    inputTh =  threading.Thread(target = detectInputKey)
    inputTh.start()

    print("Load model start")

    tflite_model = tf.contrib.lite.Interpreter(model_path="./CNN_keras_2.5s_Y50.tflite")
    tflite_model.allocate_tensors()

    print("Load model end")
    runningThreadNum = 0

    while True:
        mainLock.acquire()
        if runningThreadNum < len(deviceList):
            for i in range(len(deviceList) - runningThreadNum):
                newThread = threading.Thread(target = imuThread, args=[deviceList[-(i+1)], runningThreadNum])
                newThread.start()
                runningThreadNum = runningThreadNum + 1
        mainLock.release()

if __name__ == '__main__':
    main()




