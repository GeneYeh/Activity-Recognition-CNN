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
import scipy.interpolate

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

# need to config parameters
ower = "Gene"
activityList = ["Sit", "Lie", "Stand", "Jogging", "Walk", "Run", "Upstairs", "Downstairs", "Drink_Water", "Open_Door", "Wash_Teenth", "Stretch", "Walk_Phone"]
                #  0 ,   1    ,  2,      3   ,   4   ,   5  ,      6    ,      7      ,       8      ,      9     ,     10       ,    11
savedActivity = activityList.index("Jogging")
# long:6s, short:4s
secActivity = 6

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
        self.dataDicts= []
        self.lastTime = -1
        # for count packet number
        self.count = 0


# get rawdata from ecomini when notification
class MyDelegate(btle.DefaultDelegate):
    def __init__(self, node):
        btle.DefaultDelegate.__init__(self)
        self.node = node

    def handleNotification(self, cHandle, data):
        reading =  binascii.b2a_hex(data) # transfer string to binary then using hex format to express
        self.node.noti = Uint4Toshort([reading[0:4],reading[4:8],reading[8:12],reading[12:16],reading[16:20],reading[20:24],reading[24:28],reading[28:32],reading[32:36],reading[36:40]])
        #self.node.noti means rawdata without calibration from ecomini 
        #acc[x], acc[y], acc[z], gyro[x], gyro[y], gyro[z], mag[x], mag[y], mag[z], timer

# scan
def scanThread():
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
                        print("iface = %d" % (iface + ifaceOffset))
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
                        
                        print("accBias: ",connNode.accBias)
                        print("gyroBias: ",connNode.gyroBias)
                        print("magBias: ",connNode.magBias)
                        print("magScale: ",connNode.magScale)
                        print("magCalibration: ",connNode.magCalibration)
                        print("connect successfully")

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
                    for i, node in enumerate(deviceList):
                        print(node.accBias, "count: ", i, node.count)
                    save_data_flag="finish"

def write_to_txt(list_item, fileName):
    savedpath = 'Data' + '/' + ower + '/' + activityList[savedActivity]+ '/'
    fp = open(savedpath+fileName+'.dat', "wb")
    for item_ob in list_item:    #pickle dump
        pickle.dump(item_ob, fp)
    fp.close()
    print("finish saving file: %s" % fileName)


def eliminate_gravity(node, value):

    node.gravity[0] = alpha*node.gravity[0] + (1-alpha)*value[0]
    node.gravity[1] = alpha*node.gravity[1] + (1-alpha)*value[1]
    node.gravity[2] = alpha*node.gravity[2] + (1-alpha)*value[2]

    return value[0]-node.gravity[0], value[1]-node.gravity[1], value[2]-node.gravity[2]

def imuThread(node, i):
    global save_data_flag, savedFileNum, lock
    print("imuThread start")

    file_num = 10
    savedFlag = False

    while True:
        # try:
        if node.Peripheral.waitForNotifications(0.05):
            rawdata = node.noti
            calibratedData = dataCalibration(rawdata, i)
            axrgData, ayrgData, azrgData = eliminate_gravity(node, calibratedData[0:3])
            accRSS = calibratedData[0]**2 + calibratedData[1]**2 + calibratedData[2]**2
            gyroRSS = calibratedData[3]**2 + calibratedData[4]**2 + calibratedData[5]**2
            
            #save data per hundred packet received
            if save_data_flag == "notReady":
                notReadyFlag = True
                savedFlag = False
            if save_data_flag == "ready":
                if notReadyFlag == True:
                    notReadyFlag = False
                    node.count = 0
                dataDict = collections.OrderedDict()
                dataDict['Acc'] = mat([calibratedData[0],calibratedData[1],calibratedData[2]])
                dataDict['Gyo'] = mat([calibratedData[3],calibratedData[4],calibratedData[5]])
                dataDict['Angle'] = mat(node.nodeCube.angle)
                dataDict['timestamp'] = time.time() - tStart
                dataDict['timestampChip'] = calibratedData[6]
                dataDict['rgAcc'] = mat([axrgData,ayrgData,azrgData])
                node.dataDicts.append(dataDict)
                node.count = node.count + 1
                
        else:
            continue

        if save_data_flag == "finish" and savedFlag == False:
            savedFlag = True
            if i == 0:
                fileName = str(file_num) + 'wrist'
            elif i == 1:
                fileName = str(file_num) + 'waist'
            elif i == 2:
                fileName = str(file_num) + 'ankle'
            else:
                print("fileIndex error")
            print("start saving file: %s" % fileName)
            #print('node:', node.dataDicts)
            t =threading.Thread(target = write_to_txt, args=[node.dataDicts[:], fileName])  #[data_chunk] make  data_chunk as a arguement
            t.start()
            node.dataDicts = []
            file_num = file_num + 1
            lock.acquire()
            savedFileNum = savedFileNum + 1
            if savedFileNum == imuNodeNum:
                save_data_flag = "notReady"
                savedFileNum = 0
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

def main():
    global runningThreadNum

    scanTh =  threading.Thread(target = scanThread)
    scanTh.start()

    inputTh =  threading.Thread(target = detectInputKey)
    inputTh.start()

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


