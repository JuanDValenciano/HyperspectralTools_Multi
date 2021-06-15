#   JuanD Valenciano. jvalenciano@unal.edu.co
#   Script to radiometric convertion.
#   
#   Python3a

class UnsupportedPlatform(Exception):
    pass

import os
######
import sys
from sys import platform  #Detect platform!
#sys.path.append('/content/drive/My Drive/Tesis/DatosAnalizar/ColabLib')    #Dir Lib!
######

#from numba import jit
import time
import torch 
import numpy as np
import itertools

import spectral.io.envi as envi
from spectral import *
import warnings
import pandas as pd

###############################################################################################################################################################
if "linux" in platform:  #No sure this work on MAC?¿
    print("Linux")
    print("Colab???")
    COLAB       = 0
    LINUX_PC    = 1
    WSL         = 1&LINUX_PC
    WINDOWS_PC  = 0
elif "darwin" in platform:
    print("mac")
    print("No support!!!!!!")
    exit
elif "win" in platform:
    print("windows TAAAA :(")
    print("No support: from skimage.measure import compare_psnr")
    COLAB       = 0
    LINUX_PC    = 0
    WSL         = 1&LINUX_PC
    WINDOWS_PC  = 1
else:
    raise UnsupportedPlatform
###############################################################################################################################################################
def correctIlumNumpy( ArrayPoint, DataRef, SpectralRef, _DEBUG_ON = 0):
  #obtner el promedio de la zona que se usa como referencia.
  CounterSignalFOR = 0
  NewMean_DataTesReferenceMT3  = np.mean( DataRef[int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]),:], axis=0).astype(float)
  #NewMean_DataTesReference1 = np.mean(NewMean_DataTesReference,axis=0).astype(float) 
  if(_DEBUG_ON):
    print("> ?1600,160¡=", NewMean_DataTesReferenceMT3.shape)
  
  N_ROWS, N_COLS, N_BANDS = DataRef.shape

  ValorCoef = np.zeros((N_COLS,N_BANDS), dtype=np.float32)
  
  for BandaSeleccionada in range(N_BANDS):
    for i in range(N_COLS):
      ValorCoef[i,BandaSeleccionada] = (SpectralRef[BandaSeleccionada,1]).astype(float)/(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]).astype(float)        
      #CalibrationBands[i,BandaSeleccionada] = np.float32(ValorRef[i,BandaSeleccionada])*(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]) #SpectralReference[BandaSeleccionada,1] - np.float32(NewMean_DataTesReference[i,BandaSeleccionada] )

  NewMean_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = DataRef[:,:,:]
  valorRefCompleto = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)

  #for i in tnrange(3800, desc='??'):
  for i in range(3800):
    valorRefCompleto[i,:,:] = ValorCoef[:,:] 
  if(_DEBUG_ON):
    print(valorRefCompleto[:,:,0].shape)
    print(New_DataTesReference[:,:,0].shape)
    print(NewMean_DataTesReference[:,:,0].shape)

  for j in range(160):
    NewMean_DataTesReference[:,:,j] = np.clip( (New_DataTesReference[:,:,j] *  valorRefCompleto[:,:,j]).astype(float), a_min=0, a_max=None)
  #NewMean_DataTesReference

  return (NewMean_DataTesReference)

def correctIlumNumpyMEDIAN( ArrayPoint, DataRef, SpectralRef, _DEBUG_ON = 0):
  #obtner el promedio de la zona que se usa como referencia.
  CounterSignalFOR = 0
  NewMean_DataTesReferenceMT3  = np.median( DataRef[int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]),:], axis=0).astype(float)
  #NewMean_DataTesReference1 = np.mean(NewMean_DataTesReference,axis=0).astype(float) 
  if(_DEBUG_ON):
    print("> ?1600,160¡=", NewMean_DataTesReferenceMT3.shape)
  
  N_ROWS, N_COLS, N_BANDS = DataRef.shape

  ValorCoef = np.zeros((N_COLS,N_BANDS), dtype=np.float32)
  
  for BandaSeleccionada in range(N_BANDS):
    for i in range(N_COLS):
      ValorCoef[i,BandaSeleccionada] = (SpectralRef[BandaSeleccionada,1]).astype(float)/(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]).astype(float)
      #CalibrationBands[i,BandaSeleccionada] = np.float32(ValorRef[i,BandaSeleccionada])*(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]) #SpectralReference[BandaSeleccionada,1] - np.float32(NewMean_DataTesReference[i,BandaSeleccionada] )

  NewMean_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = DataRef[:,:,:]
  valorRefCompleto = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)

  for i in range(3800):
    valorRefCompleto[i,:,:] = ValorCoef[:,:] 
  if(_DEBUG_ON):
    print(valorRefCompleto[:,:,0].shape)
   
    print(New_DataTesReference[:,:,0].shape)
    print(NewMean_DataTesReference[:,:,0].shape)

  #for j in tnrange(160, desc='??'):
  for j in range(160):
    NewMean_DataTesReference[:,:,j] = np.clip( (New_DataTesReference[:,:,j] *  valorRefCompleto[:,:,j]).astype(float), a_min=0, a_max=None)
  #NewMean_DataTesReference

  return (NewMean_DataTesReference)
###############################################################################################################################################################
def radiometricResponseNumpy( ArrayPoint, DataRef, BlackRef, SpectralRef, IntegrationTime, _DEBUG_ON = 0):    
    # creating a rectangle
    #          Col, Row
    #Point1 = (0,100)
    #Point2 = (1600, 150)
    
    ROUND_DECIMAL = 0   # 1.4 -> 1.0  |  1.5 -> 2.0
    
    if(_DEBUG_ON):
        print("> DEBUG_ON radiometricResponseNumpy: ")
        print(ArrayPoint.shape)
  
    CounterSignal  = ArrayPoint.shape[0]
    
    if(_DEBUG_ON):
        print("CounterSignal: ")
        print(CounterSignal)
    
    WhitedataTest  = 0.0       
    WhitedataTest1 = 0.0
    WhitedataTest2 = 0.0
    WhitedataTest2_Last = 0.0
    
    for CounterSignalFOR in range(CounterSignal):
        if(_DEBUG_ON):
            print("for(Solo_Uno_?):", str(CounterSignalFOR))
        WhitedataTest  = DataRef[   int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]), :]
        #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
        #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
        WhitedataTest1 = np.mean(WhitedataTest, axis=0).astype(float)
        WhitedataTest2 = np.mean(WhitedataTest1,axis=0).astype(float)        
        if(CounterSignalFOR>0):
            if(_DEBUG_ON):
                print("NO SE DEBE DE DAR ESTE CASO........")
                print("for + SUM", str(CounterSignalFOR))
            #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
            #WhitedataTest2 = (np.round(  (WhitedataTest2+WhitedataTest2_Last)/2    ,decimals=ROUND_DECIMAL)).astype(int)
            WhitedataTest2 = np.mean(WhitedataTest1,axis=0).astype(float)
            WhitedataTest2 = ((WhitedataTest2+WhitedataTest2_Last)/2).astype(float)
        WhitedataTest  = 0.0       
        WhitedataTest1 = 0.0
        WhitedataTest2_Last = WhitedataTest2                                                                                
    if(_DEBUG_ON):
        print("EndFor")
   
    #WhitedataTest  = DataRef[ Pt1[1]:Pt2[1], Pt1[0]:Pt2[0], :]
    #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
    #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
    
    N_ROWS, N_COLS, N_BANDS = DataRef.shape
    N_ROWS = 1
    N_COLS = 1
    #NW_WhiteReferenceArray = torch.empty((N_ROWS, N_COLS, N_BANDS))
    NewRadianceCamera = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
    #for col in range(N_COLS):
    #    NewRadianceCamera[0, col, :] = (WhitedataTest2[:] - BlackRef[ 0, 0, :]) #/(SpectralRef[:,1]*IntegrationTime)    
    #NewRadianceCamera[0, 0, :] = np.clip( (WhitedataTest2[:] - BlackRef[ 0, 0, :])/(SpectralRef[:,1]*IntegrationTime), min=0)
    NewRadianceCamera[0, 0, :] = np.clip((((np.clip( WhitedataTest2[:].astype(float) - (BlackRef[ 0, 0, :]).astype(float), a_min=0, a_max=None)))/((SpectralRef[:,1]).astype(float)*IntegrationTime)), a_min=0, a_max=None) #Bug negative number!
    #NewRadianceCamera = NewRadianceCamera.clip(min=0)
    #NewRadianceCamera[0, 0, :] = (NewRadianceCamera[0, 0, :])/(SpectralRef[:,1]*IntegrationTime)
    #NewRadianceCamera = NewRadianceCamera.clip(min=0)
    
    N_ROWS, N_COLS, N_BANDS = DataRef.shape
    dataTestCalibrada = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
    '''
    for row in range(N_ROWS):
        for col in range(N_COLS):
            dataTestCalibrada[row,col,:] = np.clip(  np.clip((DataRef[row,col,:]).astype(float) - (BlackRef[0,0,:]).astype(float), a_min=0, a_max=None)   /(NewRadianceCamera[0,0,:]*IntegrationTime), a_min=0, a_max=None)
    '''
    for row in range(N_ROWS):
        for col in range(N_COLS):
            dataTestCalibrada[row,col,:] = np.clip(  np.clip((DataRef[row,col,:]).astype(float) - (BlackRef[0,0,:]).astype(float), a_min=0, a_max=None)/(NewRadianceCamera[0,0,:]*IntegrationTime), a_min=0, a_max=None)
    
    #dataTestCalibrada = dataTestCalibrada.clip(min=0)
    #print(WhitedataTest.shape)
    #print(WhitedataTest.shape[2])
    #print(WhitedataTest1.shape)
    #print(WhitedataTest2.shape)
    #Reference1 = carlos_3_8_16000_17000__16000_us_2x_2020_01_27T202006_corr[:,:,:]
    #Reference2 = (np.round(Reference1.sum(axis=0)/carlos_3_8_16000_17000__16000_us_2x_2020_01_27T202006_corr.nrows ,decimals=ROUND_DECIMAL)).astype(int)
    #Reference5 = (np.round(Reference2.sum(axis=0)/carlos_3_8_16000_17000__16000_us_2x_2020_01_27T202006_corr.ncols ,decimals=ROUND_DECIMAL)).astype(int)
    ##return (NewRadianceCamera)
    
    #del dataTestCalibrada
    del NewRadianceCamera
    del WhitedataTest
    del WhitedataTest1
    del WhitedataTest2_Last
    del WhitedataTest2 
    #dataTestCalibrada = 0
    # Remove a reference to original data:
    #dataTestCalibrada.data = None
    return (dataTestCalibrada)

def radiometricResponseNumpyMEDIAN( ArrayPoint, DataRef, BlackRef, SpectralRef, IntegrationTime, _DEBUG_ON = 0):    
    # creating a rectangle
    #          Col, Row
    #Point1 = (0,100)
    #Point2 = (1600, 150)
    
    ROUND_DECIMAL = 0   # 1.4 -> 1.0  |  1.5 -> 2.0
    
    if(_DEBUG_ON):
        print("Function radiometricResponseNumpy: ")
        print(ArrayPoint.shape)
  
    CounterSignal  = ArrayPoint.shape[0]
    if(_DEBUG_ON):
        print("CounterSignal: ")
        print(CounterSignal)
    
    WhitedataTest  = 0       
    WhitedataTest1 = 0
    WhitedataTest2 = 0
    WhitedataTest2_Last = 0
    
    for CounterSignalFOR in range(CounterSignal):
        if(_DEBUG_ON):
            print("for + ", str(CounterSignalFOR))
        WhitedataTest  = DataRef[   int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]), :]
        #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
        #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
        WhitedataTest1 = np.median(WhitedataTest, axis=0).astype(float)
        WhitedataTest2 = np.median(WhitedataTest1,axis=0).astype(float)        
        if(CounterSignalFOR>0):
            if(_DEBUG_ON):
                print("for + SUM", str(CounterSignalFOR))
            #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
            #WhitedataTest2 = (np.round(  (WhitedataTest2+WhitedataTest2_Last)/2    ,decimals=ROUND_DECIMAL)).astype(int)
            WhitedataTest2 = np.median(WhitedataTest1,axis=0).astype(float)
            WhitedataTest2 = ((WhitedataTest2+WhitedataTest2_Last)/2).astype(float)
        WhitedataTest  = 0       
        WhitedataTest1 = 0
        WhitedataTest2_Last = WhitedataTest2                                                                                
    if(_DEBUG_ON):
        print("EndFor")
    
    #WhitedataTest  = DataRef[ Pt1[1]:Pt2[1], Pt1[0]:Pt2[0], :]
    #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
    #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
    
    N_ROWS, N_COLS, N_BANDS = DataRef.shape
    N_ROWS = 1
    N_COLS = 1
    #NW_WhiteReferenceArray = torch.empty((N_ROWS, N_COLS, N_BANDS))
    NewRadianceCamera = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
    #for col in range(N_COLS):
    #    NewRadianceCamera[0, col, :] = (WhitedataTest2[:] - BlackRef[ 0, 0, :]) #/(SpectralRef[:,1]*IntegrationTime)    
    #NewRadianceCamera[0, 0, :] = np.clip( (WhitedataTest2[:] - BlackRef[ 0, 0, :])/(SpectralRef[:,1]*IntegrationTime), min=0)
    NewRadianceCamera[0, 0, :] = np.clip((((np.clip( WhitedataTest2[:].astype(float) - (BlackRef[ 0, 0, :]).astype(float), a_min=0, a_max=None)))/((SpectralRef[:,1]).astype(float)*IntegrationTime)), a_min=0, a_max=None) #Bug negative number!
    #NewRadianceCamera = NewRadianceCamera.clip(min=0)
    #NewRadianceCamera[0, 0, :] = (NewRadianceCamera[0, 0, :])/(SpectralRef[:,1]*IntegrationTime)
    #NewRadianceCamera = NewRadianceCamera.clip(min=0)
    
    N_ROWS, N_COLS, N_BANDS = DataRef.shape
    dataTestCalibrada = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
    
    for row in range(N_ROWS):
        for col in range(N_COLS):
            dataTestCalibrada[row,col,:] = np.clip(  (np.clip((DataRef[row,col,:]).astype(float) - (BlackRef[0,0,:]).astype(float), a_min=0, a_max=None)/(NewRadianceCamera[0,0,:]*IntegrationTime)), a_min=0, a_max=None)
    
    #dataTestCalibrada = dataTestCalibrada.clip(min=0)
    #print(WhitedataTest.shape)
    #print(WhitedataTest.shape[2])
    #print(WhitedataTest1.shape)
    #print(WhitedataTest2.shape)
    #Reference1 = carlos_3_8_16000_17000__16000_us_2x_2020_01_27T202006_corr[:,:,:]
    #Reference2 = (np.round(Reference1.sum(axis=0)/carlos_3_8_16000_17000__16000_us_2x_2020_01_27T202006_corr.nrows ,decimals=ROUND_DECIMAL)).astype(int)
    #Reference5 = (np.round(Reference2.sum(axis=0)/carlos_3_8_16000_17000__16000_us_2x_2020_01_27T202006_corr.ncols ,decimals=ROUND_DECIMAL)).astype(int)
    ##return (NewRadianceCamera)
    del NewRadianceCamera
    del WhitedataTest
    del WhitedataTest1
    del WhitedataTest2_Last
    del WhitedataTest2
    return (dataTestCalibrada)
###############################################################################################################################################################
def correctIlumNumpy_OtherPath( ArrayPoint, DataRef, ArrayPointApply, SpectralRef, _DEBUG_ON = 0):
  #obtner el promedio de la zona que se usa como referencia.
  CounterSignalFOR = 0
  NewMean_DataTesReferenceMT3  = np.mean( DataRef[int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]),:], axis=0).astype(float)
  #NewMean_DataTesReference1 = np.mean(NewMean_DataTesReference,axis=0).astype(float) 
  if(_DEBUG_ON):
    print("> ?1600,160¡=", NewMean_DataTesReferenceMT3.shape)
  
  N_ROWS, N_COLS, N_BANDS = DataRef.shape

  ValorCoef = np.zeros((N_COLS,N_BANDS), dtype=np.float32)
  
  for BandaSeleccionada in range(N_BANDS):
    for i in range(N_COLS):
      ValorCoef[i,BandaSeleccionada] = (SpectralRef[BandaSeleccionada,1]).astype(float)/(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]).astype(float)
      #CalibrationBands[i,BandaSeleccionada] = np.float32(ValorRef[i,BandaSeleccionada])*(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]) #SpectralReference[BandaSeleccionada,1] - np.float32(NewMean_DataTesReference[i,BandaSeleccionada] )

  NewMean_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = ArrayPointApply[:,:,:] #Mod la otra imagen para tener en cuenta.
  valorRefCompleto = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)

  for i in tnrange(3800, desc='??'):
    valorRefCompleto[i,:,:] = ValorCoef[:,:] 
  if(_DEBUG_ON):
    print(valorRefCompleto[:,:,0].shape)
    print(New_DataTesReference[:,:,0].shape)
    print(NewMean_DataTesReference[:,:,0].shape)

  for j in tnrange(160, desc='??'):
    NewMean_DataTesReference[:,:,j] = np.clip( (New_DataTesReference[:,:,j] *  valorRefCompleto[:,:,j]).astype(float), a_min=0, a_max=None)
  #NewMean_DataTesReference

  return (NewMean_DataTesReference)

def correctIlumNumpyMEDIAN_OtherPath(ArrayPoint, DataRef, ArrayPointApply, SpectralRef, _DEBUG_ON = 0):
  #obtner el promedio de la zona que se usa como referencia.
  CounterSignalFOR = 0
  NewMean_DataTesReferenceMT3  = np.median( DataRef[int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]),:], axis=0).astype(float)
  #NewMean_DataTesReference1 = np.mean(NewMean_DataTesReference,axis=0).astype(float) 
  if(_DEBUG_ON):
    print("> ?1600,160¡=", NewMean_DataTesReferenceMT3.shape)
  
  N_ROWS, N_COLS, N_BANDS = DataRef.shape

  ValorCoef = np.zeros((N_COLS,N_BANDS), dtype=np.float32)
  
  for BandaSeleccionada in range(N_BANDS):
    for i in range(N_COLS):
      ValorCoef[i,BandaSeleccionada] = (SpectralRef[BandaSeleccionada,1]).astype(float)/(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]).astype(float)
      #CalibrationBands[i,BandaSeleccionada] = np.float32(ValorRef[i,BandaSeleccionada])*(NewMean_DataTesReferenceMT3[i,BandaSeleccionada]) #SpectralReference[BandaSeleccionada,1] - np.float32(NewMean_DataTesReference[i,BandaSeleccionada] )

  NewMean_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)
  New_DataTesReference = ArrayPointApply[:,:,:]
  valorRefCompleto = np.zeros((N_ROWS,N_COLS,N_BANDS), dtype=np.float32)

  for i in tnrange(3800, desc='??'):
    valorRefCompleto[i,:,:] = ValorCoef[:,:] 
  if(_DEBUG_ON):
    print(valorRefCompleto[:,:,0].shape)
    print(New_DataTesReference[:,:,0].shape)
    print(NewMean_DataTesReference[:,:,0].shape)

  for j in tnrange(160, desc='??'):
    NewMean_DataTesReference[:,:,j] = np.clip( (New_DataTesReference[:,:,j] *  valorRefCompleto[:,:,j]).astype(float), a_min=0, a_max=None)
  #NewMean_DataTesReference

  return (NewMean_DataTesReference)
###############################################################################################################################################################
def radiometricResponseNumpy_OtherPath( ArrayPoint, DataRef, ArrayPointApply, BlackRef, SpectralRef, IntegrationTime, _DEBUG_ON = 0):  
  # creating a rectangle
  #          Col, Row
  #Point1 = (0,100)
  #Point2 = (1600, 150)
  
  ROUND_DECIMAL = 0   # 1.4 -> 1.0  |  1.5 -> 2.0
  
  if(_DEBUG_ON):
      print("> DEBUG_ON radiometricResponseNumpy: ")
      print(ArrayPoint.shape)

  CounterSignal  = ArrayPoint.shape[0]
  
  if(_DEBUG_ON):
      print("CounterSignal: ")
      print(CounterSignal)
  
  WhitedataTest  = 0.0       
  WhitedataTest1 = 0.0
  WhitedataTest2 = 0.0
  WhitedataTest2_Last = 0.0
  
  for CounterSignalFOR in range(CounterSignal):
      if(_DEBUG_ON):
          print("for(Solo_Uno_?):", str(CounterSignalFOR))
      WhitedataTest  = DataRef[   int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]), :]
      #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
      #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
      WhitedataTest1 = np.mean(WhitedataTest, axis=0).astype(float)
      WhitedataTest2 = np.mean(WhitedataTest1,axis=0).astype(float)        
      if(CounterSignalFOR>0):
          if(_DEBUG_ON):
              print("NO SE DEBE DE DAR ESTE CASO........")
              print("for + SUM", str(CounterSignalFOR))
          #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
          #WhitedataTest2 = (np.round(  (WhitedataTest2+WhitedataTest2_Last)/2    ,decimals=ROUND_DECIMAL)).astype(int)
          WhitedataTest2 = np.mean(WhitedataTest1,axis=0).astype(float)
          WhitedataTest2 = ((WhitedataTest2+WhitedataTest2_Last)/2).astype(float)
      WhitedataTest  = 0.0       
      WhitedataTest1 = 0.0
      WhitedataTest2_Last = WhitedataTest2                                                                                
  if(_DEBUG_ON):
      print("EndFor")
  
  #WhitedataTest  = DataRef[ Pt1[1]:Pt2[1], Pt1[0]:Pt2[0], :]
  #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
  #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
  
  N_ROWS, N_COLS, N_BANDS = DataRef.shape
  N_ROWS = 1
  N_COLS = 1
  #NW_WhiteReferenceArray = torch.empty((N_ROWS, N_COLS, N_BANDS))
  NewRadianceCamera = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
  #for col in range(N_COLS):
  #    NewRadianceCamera[0, col, :] = (WhitedataTest2[:] - BlackRef[ 0, 0, :]) #/(SpectralRef[:,1]*IntegrationTime)    
  #NewRadianceCamera[0, 0, :] = np.clip( (WhitedataTest2[:] - BlackRef[ 0, 0, :])/(SpectralRef[:,1]*IntegrationTime), min=0)
  NewRadianceCamera[0, 0, :] = np.clip((((np.clip( WhitedataTest2[:].astype(float) - (BlackRef[ 0, 0, :]).astype(float), a_min=0, a_max=None)))/((SpectralRef[:,1]).astype(float)*IntegrationTime)), a_min=0, a_max=None) #Bug negative number!
  #NewRadianceCamera = NewRadianceCamera.clip(min=0)
  #NewRadianceCamera[0, 0, :] = (NewRadianceCamera[0, 0, :])/(SpectralRef[:,1]*IntegrationTime)
  #NewRadianceCamera = NewRadianceCamera.clip(min=0)
  
  N_ROWS, N_COLS, N_BANDS = ArrayPointApply.shape
  dataTestCalibrada = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
  
  for row in range(N_ROWS):
      for col in range(N_COLS):
          dataTestCalibrada[row,col,:] = np.clip(  np.clip((ArrayPointApply[row,col,:]).astype(float) - (BlackRef[0,0,:]).astype(float), a_min=0, a_max=None)/(NewRadianceCamera[0,0,:]*IntegrationTime), a_min=0, a_max=None)
  
  #del dataTestCalibrada
  del NewRadianceCamera
  del WhitedataTest
  del WhitedataTest1
  del WhitedataTest2_Last
  del WhitedataTest2 
  #dataTestCalibrada = 0
  # Remove a reference to original data:
  #dataTestCalibrada.data = None
  return (dataTestCalibrada)

def radiometricResponseNumpyMEDIAN_OtherPath( ArrayPoint, DataRef, ArrayPointApply, BlackRef, SpectralRef, IntegrationTime, _DEBUG_ON = 0):  
  # creating a rectangle
  #          Col, Row
  #Point1 = (0,100)
  #Point2 = (1600, 150)
  
  ROUND_DECIMAL = 0   # 1.4 -> 1.0  |  1.5 -> 2.0
  
  if(_DEBUG_ON):
      print("> DEBUG_ON radiometricResponseNumpy: ")
      print(ArrayPoint.shape)

  CounterSignal  = ArrayPoint.shape[0]
  
  if(_DEBUG_ON):
      print("CounterSignal: ")
      print(CounterSignal)
  
  WhitedataTest  = 0.0       
  WhitedataTest1 = 0.0
  WhitedataTest2 = 0.0
  WhitedataTest2_Last = 0.0
  
  for CounterSignalFOR in range(CounterSignal):
      if(_DEBUG_ON):
          print("for(Solo_Uno_?):", str(CounterSignalFOR))
      WhitedataTest  = DataRef[   int(ArrayPoint[ int(CounterSignalFOR),0]): int(ArrayPoint[ int(CounterSignalFOR),2]),  int(ArrayPoint[ int(CounterSignalFOR),1]):int(ArrayPoint[ int(CounterSignalFOR),3]), :]
      #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
      #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
      WhitedataTest1 = np.median(WhitedataTest, axis=0).astype(float)
      WhitedataTest2 = np.median(WhitedataTest1,axis=0).astype(float)        
      if(CounterSignalFOR>0):
          if(_DEBUG_ON):
              print("NO SE DEBE DE DAR ESTE CASO........")
              print("for + SUM", str(CounterSignalFOR))
          #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int)
          #WhitedataTest2 = (np.round(  (WhitedataTest2+WhitedataTest2_Last)/2    ,decimals=ROUND_DECIMAL)).astype(int)
          WhitedataTest2 = np.median(WhitedataTest1,axis=0).astype(float)
          WhitedataTest2 = ((WhitedataTest2+WhitedataTest2_Last)/2).astype(float)
      WhitedataTest  = 0.0       
      WhitedataTest1 = 0.0
      WhitedataTest2_Last = WhitedataTest2                                                                                
  if(_DEBUG_ON):
      print("EndFor")
  
  #WhitedataTest  = DataRef[ Pt1[1]:Pt2[1], Pt1[0]:Pt2[0], :]
  #WhitedataTest1 = (np.round(WhitedataTest.sum(axis=0)/WhitedataTest.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
  #WhitedataTest2 = (np.round(WhitedataTest1.sum(axis=0)/WhitedataTest1.shape[0] ,decimals=ROUND_DECIMAL)).astype(int) 
  
  N_ROWS, N_COLS, N_BANDS = DataRef.shape
  N_ROWS = 1
  N_COLS = 1
  #NW_WhiteReferenceArray = torch.empty((N_ROWS, N_COLS, N_BANDS))
  NewRadianceCamera = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
  #for col in range(N_COLS):
  #    NewRadianceCamera[0, col, :] = (WhitedataTest2[:] - BlackRef[ 0, 0, :]) #/(SpectralRef[:,1]*IntegrationTime)    
  #NewRadianceCamera[0, 0, :] = np.clip( (WhitedataTest2[:] - BlackRef[ 0, 0, :])/(SpectralRef[:,1]*IntegrationTime), min=0)
  NewRadianceCamera[0, 0, :] = np.clip((((np.clip( WhitedataTest2[:].astype(float) - (BlackRef[ 0, 0, :]).astype(float), a_min=0, a_max=None)))/((SpectralRef[:,1]).astype(float)*IntegrationTime)), a_min=0, a_max=None) #Bug negative number!
  #NewRadianceCamera = NewRadianceCamera.clip(min=0)
  #NewRadianceCamera[0, 0, :] = (NewRadianceCamera[0, 0, :])/(SpectralRef[:,1]*IntegrationTime)
  #NewRadianceCamera = NewRadianceCamera.clip(min=0)
  
  N_ROWS, N_COLS, N_BANDS = ArrayPointApply.shape
  dataTestCalibrada = np.zeros((N_ROWS, N_COLS, N_BANDS), dtype=np.float32)
  
  for row in tnrange(N_ROWS):
      for col in range(N_COLS):
          dataTestCalibrada[row,col,:] = np.clip(  np.clip((ArrayPointApply[row,col,:]).astype(float) - (BlackRef[0,0,:]).astype(float), a_min=0, a_max=None)/(NewRadianceCamera[0,0,:]*IntegrationTime), a_min=0, a_max=None)
    
  #del dataTestCalibrada
  del NewRadianceCamera
  del WhitedataTest
  del WhitedataTest1
  del WhitedataTest2_Last
  del WhitedataTest2 
  #dataTestCalibrada = 0
  # Remove a reference to original data:
  #dataTestCalibrada.data = None
  return (dataTestCalibrada)
###############################################################################################################################################################
#Datos Almacenados en WS
# Black Reference and spectral target.
if(COLAB):
    BlackReference_PATH     = '/content/drive/MyDrive/Tesis/DatosAnalizar/Experiment 2/BlackReference/BlackReference_Single'
    spectral_target_PATH    = '/content/drive/MyDrive/Tesis/software/spectra_target'
elif(LINUX_PC&WSL):
    BlackReference_PATH     = '/mnt/c/Users/Desarrollo/Ubuntu_Folder/ToolsBlackREF_HIS/BlackReference/BlackReference_Single' 
    spectral_target_PATH    = '/mnt/c/Users/Desarrollo/Ubuntu_Folder/ToolsBlackREF_HIS/spectra_target'
elif(LINUX_PC):
    print(">!Solo Linux!!!!!!!")
    BlackReference_PATH     = ''
    spectral_target_PATH    = ''
elif(WINDOWS_PC):
    print(">!CargarpathWindows!!!!!!!")
    BlackReference_PATH     = '/Users/Desarrollo/Ubuntu_Folder/ToolsBlackREF_HIS/BlackReference/BlackReference_Single' 
    spectral_target_PATH    = '/Users/Desarrollo/Ubuntu_Folder/ToolsBlackREF_HIS/spectra_target'
else:
    print(">!Envoroment")

BLACK_REF_IMG      = envi.open( BlackReference_PATH + '.hdr', BlackReference_PATH + '.hyspex')
dataSpectralTarget = pd.read_csv( spectral_target_PATH + '.csv',   header=None)

#Spectral reference.
N_COLS  = 2
N_ROWS  = 160

SpectralReference = np.empty((N_ROWS, N_COLS))
SpectralReference[:,0] = dataSpectralTarget[0] #Copy wn
SpectralReference[:,1] = dataSpectralTarget[1] #Copy Radiance

IntegrationTime = 16000
##############################################################################################################################################################

# Comprobación de seguridad, ejecutar sólo si se reciben 2 
# argumentos realemente

#example use RAD: 
#python ForColab.py RAD MEAN C:\Users\Desarrollo\Ubuntu_Folder\ExperimentHyspex\Fairchild_inoculate_sample3\fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 C:\Users\Desarrollo\Ubuntu_Folder\ExperimentHyspex\SaveTest\

#example use RAD+ILU: 
#python TT.py RAD+ILU MEAN C:\Users\Desarrollo\Ubuntu_Folder\ExperimentHyspex\Fairchild_inoculate_sample3\fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 C:\Users\Desarrollo\Ubuntu_Folder\ExperimentHyspex\SaveTest\

####################### On Linux #################
#example use RAD:            
#python ForColab.py RAD MEAN /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/SaveTest/

#example use RAD+ILU:            
#python ForColab.py RAD+ILU MEAN /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/SaveTest/

#example use ILU: 
#python ForColab.py ILU MEAN /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/SaveTest/

#example use RAD_OTH: 
#python ForColab.py RAD_OTH MEAN /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/SaveTest/

#example use ILU_OTH: 
#python ForColab.py ILU_OTH MEAN /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/SaveTest/

#example use RAD+ILU_OTH: 
#python ForColab.py RAD+ILU_OTH MEAN /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/Fairchild_inoculate_sample3/fai_igs_03_16000_us_2x_2019-11-24T123406_corr 0 100 /mnt/c/Users/Desarrollo/Ubuntu_Folder/ExperimentHyspex/SaveTest/

_DEBUG_ON = 1

_MOD = sys.argv[1]

if(_MOD == "RAD"):
    if( len(sys.argv) == 7):
        _ALG = sys.argv[2]
        _FILE_PATH = sys.argv[3]
        _MIN = int(sys.argv[4])
        _MAX = int(sys.argv[5])
        _SAVE_PATH = sys.argv[6]
        if(_DEBUG_ON):
            print("Recv _MOD: ", _MOD)
            print("Recv _ALG: ", _ALG)
            print("Recv _FILE_PATH: ", _FILE_PATH)
            print("Recv _MIN: ", _MIN)
            print("Recv _MAX: ", _MAX)
            print("Recv _SAVE_PATH: ", _SAVE_PATH)
            #Check \ on the end save Path!!!!!!
            print("Send EndSave \:", _SAVE_PATH[len(_SAVE_PATH)-1])

        #if(_SAVE_PATH[len(_SAVE_PATH)-1] != "\\" & _SAVE_PATH[len(_SAVE_PATH)-1] != "/"):
        #    print("................ERROR................")
        #    print("PATH SAVE Folder no use \ and the end.")
        #    exit(0)
        print(_FILE_PATH + '.hdr')
        DataRecv = envi.open( _FILE_PATH + '.hdr', _FILE_PATH + '.hyspex')

        ArrayCalPoint = np.zeros(( int(1),4))
        ArrayCalPoint[0, 0] = _MIN
        ArrayCalPoint[0, 1] = 0
        ArrayCalPoint[0, 2] = _MAX
        ArrayCalPoint[0, 3] = 1600

        n_rows  =   DataRecv.nrows 
        n_cols  =   DataRecv.ncols 
        n_bands =   DataRecv.nbands

        static_hdr = {
                      'lines': n_rows,
                      'samples': n_cols,
                      'bands': n_bands,
                      'header offset': 0,
                      'data type': 4,
                      'data ignore value': 2,
                      'default bands' : [55,41,12],
                      'byte order': 0,
                      'wavelength units': "nm",
                      'wavelength': [417.563436, 421.192819, 424.822201, 428.451583, 432.080965, 435.710347, 439.339730, 442.969112, 446.598494, 450.227876, 453.857258, 457.486640, 461.116023, 464.745405, 468.374787, 472.004169, 475.633551, 479.262934, 482.892316, 486.521698, 490.151080, 493.780462, 497.409844, 501.039227, 504.668609, 508.297991, 511.927373, 515.556755, 519.186138, 522.815520, 526.444902, 530.074284, 533.703666, 537.333048, 540.962431, 544.591813, 548.221195, 551.850577, 555.479959, 559.109341, 562.738724, 566.368106, 569.997488, 573.626870, 577.256252, 580.885635, 584.515017, 588.144399, 591.773781, 595.403163, 599.032545, 602.661928, 606.291310, 609.920692, 613.550074, 617.179456, 620.808839, 624.438221, 628.067603, 631.696985, 635.326367, 638.955749, 642.585132, 646.214514, 649.843896, 653.473278, 657.102660, 660.732043, 664.361425, 667.990807, 671.620189, 675.249571, 678.878953, 682.508336, 686.137718, 689.767100, 693.396482, 697.025864, 700.655247, 704.284629, 707.914011, 711.543393, 715.172775, 718.802157, 722.431540, 726.060922, 729.690304, 733.319686, 736.949068, 740.578451, 744.207833, 747.837215, 751.466597, 755.095979, 758.725361, 762.354744, 765.984126, 769.613508, 773.242890, 776.872272, 780.501655, 784.131037, 787.760419, 791.389801, 795.019183, 798.648565, 802.277948, 805.907330, 809.536712, 813.166094, 816.795476, 820.424858, 824.054241, 827.683623, 831.313005, 834.942387, 838.571769, 842.201152, 845.830534, 849.459916, 853.089298, 856.718680, 860.348062, 863.977445, 867.606827, 871.236209, 874.865591, 878.494973, 882.124356, 885.753738, 889.383120, 893.012502, 896.641884, 900.271266, 903.900649, 907.530031, 911.159413, 914.788795, 918.418177, 922.047560, 925.676942, 929.306324, 932.935706, 936.565088, 940.194470, 943.823853, 947.453235, 951.082617, 954.711999, 958.341381, 961.970764, 965.600146, 969.229528, 972.858910, 976.488292, 980.117674, 983.747057, 987.376439, 991.005821, 994.635203]
        }
        
        #Name process convertion.
        _NAME_PROCESS = '_RAD'

        #Extraer el nombre del archivo segun la ruta asiganada.
        nameFile2Create = _FILE_PATH.split('/')
        if(_DEBUG_ON):
            print("Procces to delete / ")
            #print('OK: ', entry2.strip('hdr').strip('.').lstrip())
            print(nameFile2Create)
            print( nameFile2Create[len(nameFile2Create)-1] )
            print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
        if(not _DEBUG_ON):
            if(_ALG == "MEAN"):
                Data2Save = radiometricResponseNumpy( ArrayCalPoint, DataRecv[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print('>Save RAD MEAN')
                print('>SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            elif(_ALG == "MEDIAN"):
                Data2Save = radiometricResponseNumpyMEDIAN( ArrayCalPoint, DataRecv[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print('>Save RAD MEDIAN')
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            else:
                print("No define. Posible ERROR.!!!!!!!!!!!!!")
        else:
            print("If not Error Not procces data!!! Check the _DEBUG_ON")

    else:
        print("ERROR define number argument")
elif(_MOD == "RAD+ILU"):
    if( len(sys.argv) == 7):
        _ALG = sys.argv[2]
        _FILE_PATH = sys.argv[3]
        _MIN = int(sys.argv[4])
        _MAX = int(sys.argv[5])
        _SAVE_PATH = sys.argv[6]
        if(_DEBUG_ON):
            print("Recv _MOD: ", _MOD)
            print("Recv _ALG: ", _ALG)
            print("Recv _FILE_PATH: ", _FILE_PATH)
            print("Recv _MIN: ", _MIN)
            print("Recv _MAX: ", _MAX)
            print("Recv _SAVE_PATH: ", _SAVE_PATH)
            #Check \ on the end save Path!!!!!!
            print("Send EndSave \:", _SAVE_PATH[len(_SAVE_PATH)-1])

        #if(_SAVE_PATH[len(_SAVE_PATH)-1] != "\\" & _SAVE_PATH[len(_SAVE_PATH)-1] != "/"):
        #    print("................ERROR................")
        #    print("PATH SAVE Folder no use \ and the end.")
        #    exit(0)
        print(_FILE_PATH + '.hdr')
        DataRecv = envi.open( _FILE_PATH + '.hdr', _FILE_PATH + '.hyspex')

        ArrayCalPoint = np.zeros(( int(1),4))
        ArrayCalPoint[0, 0] = _MIN
        ArrayCalPoint[0, 1] = 0
        ArrayCalPoint[0, 2] = _MAX
        ArrayCalPoint[0, 3] = 1600

        n_rows  =   DataRecv.nrows 
        n_cols  =   DataRecv.ncols 
        n_bands =   DataRecv.nbands

        static_hdr = {
                      'lines': n_rows,
                      'samples': n_cols,
                      'bands': n_bands,
                      'header offset': 0,
                      'data type': 4,
                      'data ignore value': 2,
                      'default bands' : [55,41,12],
                      'byte order': 0,
                      'wavelength units': "nm",
                      'wavelength': [417.563436, 421.192819, 424.822201, 428.451583, 432.080965, 435.710347, 439.339730, 442.969112, 446.598494, 450.227876, 453.857258, 457.486640, 461.116023, 464.745405, 468.374787, 472.004169, 475.633551, 479.262934, 482.892316, 486.521698, 490.151080, 493.780462, 497.409844, 501.039227, 504.668609, 508.297991, 511.927373, 515.556755, 519.186138, 522.815520, 526.444902, 530.074284, 533.703666, 537.333048, 540.962431, 544.591813, 548.221195, 551.850577, 555.479959, 559.109341, 562.738724, 566.368106, 569.997488, 573.626870, 577.256252, 580.885635, 584.515017, 588.144399, 591.773781, 595.403163, 599.032545, 602.661928, 606.291310, 609.920692, 613.550074, 617.179456, 620.808839, 624.438221, 628.067603, 631.696985, 635.326367, 638.955749, 642.585132, 646.214514, 649.843896, 653.473278, 657.102660, 660.732043, 664.361425, 667.990807, 671.620189, 675.249571, 678.878953, 682.508336, 686.137718, 689.767100, 693.396482, 697.025864, 700.655247, 704.284629, 707.914011, 711.543393, 715.172775, 718.802157, 722.431540, 726.060922, 729.690304, 733.319686, 736.949068, 740.578451, 744.207833, 747.837215, 751.466597, 755.095979, 758.725361, 762.354744, 765.984126, 769.613508, 773.242890, 776.872272, 780.501655, 784.131037, 787.760419, 791.389801, 795.019183, 798.648565, 802.277948, 805.907330, 809.536712, 813.166094, 816.795476, 820.424858, 824.054241, 827.683623, 831.313005, 834.942387, 838.571769, 842.201152, 845.830534, 849.459916, 853.089298, 856.718680, 860.348062, 863.977445, 867.606827, 871.236209, 874.865591, 878.494973, 882.124356, 885.753738, 889.383120, 893.012502, 896.641884, 900.271266, 903.900649, 907.530031, 911.159413, 914.788795, 918.418177, 922.047560, 925.676942, 929.306324, 932.935706, 936.565088, 940.194470, 943.823853, 947.453235, 951.082617, 954.711999, 958.341381, 961.970764, 965.600146, 969.229528, 972.858910, 976.488292, 980.117674, 983.747057, 987.376439, 991.005821, 994.635203]
        }
        
        #Name process convertion.
        _NAME_PROCESS = '_RAD+ILU'

        #Extraer el nombre del archivo segun la ruta asiganada.
        nameFile2Create = _FILE_PATH.split('/')
        if(_DEBUG_ON):
            print("Procces to delete / ")
            #print('OK: ', entry2.strip('hdr').strip('.').lstrip())
            print(nameFile2Create)
            print( nameFile2Create[len(nameFile2Create)-1] )
            print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
        
        if(not _DEBUG_ON):
            if(_ALG == "MEAN"):
                Data2Save = correctIlumNumpy( ArrayCalPoint, radiometricResponseNumpy( ArrayCalPoint, DataRecv[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1), SpectralReference,1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print('Save RAD+ILU MEAN')
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            elif(_ALG == "MEDIAN"):
                Data2Save = correctIlumNumpyMEDIAN( ArrayCalPoint, radiometricResponseNumpyMEDIAN( ArrayCalPoint, DataRecv[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1), SpectralReference,1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print('Save RAD+ILU MEDIAN')
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            else:
                print("No define. Posible ERROR.!!!!!!!!!!!!!")
        else:
            print("If not Error Not procces data!!! Check the _DEBUG_ON")

    else:
        print("ERROR define number argument")
elif(_MOD == "ILU"):
    if( len(sys.argv) == 7):
        _ALG = sys.argv[2]
        _FILE_PATH = sys.argv[3]
        _MIN = int(sys.argv[4])
        _MAX = int(sys.argv[5])
        _SAVE_PATH = sys.argv[6]
        if(_DEBUG_ON):
            print("Recv _MOD: ", _MOD)
            print("Recv _ALG: ", _ALG)
            print("Recv _FILE_PATH: ", _FILE_PATH)
            print("Recv _MIN: ", _MIN)
            print("Recv _MAX: ", _MAX)
            print("Recv _SAVE_PATH: ", _SAVE_PATH)
            #Check \ on the end save Path!!!!!!
            print("Send EndSave \:", _SAVE_PATH[len(_SAVE_PATH)-1])
        print(_FILE_PATH + '.hdr')
        DataRecv = envi.open( _FILE_PATH + '.hdr', _FILE_PATH + '.hyspex')

        ArrayCalPoint = np.zeros(( int(1),4))
        ArrayCalPoint[0, 0] = _MIN
        ArrayCalPoint[0, 1] = 0
        ArrayCalPoint[0, 2] = _MAX
        ArrayCalPoint[0, 3] = 1600

        n_rows  =   DataRecv.nrows 
        n_cols  =   DataRecv.ncols 
        n_bands =   DataRecv.nbands

        static_hdr = {
                      'lines': n_rows,
                      'samples': n_cols,
                      'bands': n_bands,
                      'header offset': 0,
                      'data type': 4,
                      'data ignore value': 2,
                      'default bands' : [55,41,12],
                      'byte order': 0,
                      'wavelength units': "nm",
                      'wavelength': [417.563436, 421.192819, 424.822201, 428.451583, 432.080965, 435.710347, 439.339730, 442.969112, 446.598494, 450.227876, 453.857258, 457.486640, 461.116023, 464.745405, 468.374787, 472.004169, 475.633551, 479.262934, 482.892316, 486.521698, 490.151080, 493.780462, 497.409844, 501.039227, 504.668609, 508.297991, 511.927373, 515.556755, 519.186138, 522.815520, 526.444902, 530.074284, 533.703666, 537.333048, 540.962431, 544.591813, 548.221195, 551.850577, 555.479959, 559.109341, 562.738724, 566.368106, 569.997488, 573.626870, 577.256252, 580.885635, 584.515017, 588.144399, 591.773781, 595.403163, 599.032545, 602.661928, 606.291310, 609.920692, 613.550074, 617.179456, 620.808839, 624.438221, 628.067603, 631.696985, 635.326367, 638.955749, 642.585132, 646.214514, 649.843896, 653.473278, 657.102660, 660.732043, 664.361425, 667.990807, 671.620189, 675.249571, 678.878953, 682.508336, 686.137718, 689.767100, 693.396482, 697.025864, 700.655247, 704.284629, 707.914011, 711.543393, 715.172775, 718.802157, 722.431540, 726.060922, 729.690304, 733.319686, 736.949068, 740.578451, 744.207833, 747.837215, 751.466597, 755.095979, 758.725361, 762.354744, 765.984126, 769.613508, 773.242890, 776.872272, 780.501655, 784.131037, 787.760419, 791.389801, 795.019183, 798.648565, 802.277948, 805.907330, 809.536712, 813.166094, 816.795476, 820.424858, 824.054241, 827.683623, 831.313005, 834.942387, 838.571769, 842.201152, 845.830534, 849.459916, 853.089298, 856.718680, 860.348062, 863.977445, 867.606827, 871.236209, 874.865591, 878.494973, 882.124356, 885.753738, 889.383120, 893.012502, 896.641884, 900.271266, 903.900649, 907.530031, 911.159413, 914.788795, 918.418177, 922.047560, 925.676942, 929.306324, 932.935706, 936.565088, 940.194470, 943.823853, 947.453235, 951.082617, 954.711999, 958.341381, 961.970764, 965.600146, 969.229528, 972.858910, 976.488292, 980.117674, 983.747057, 987.376439, 991.005821, 994.635203]
        }       

        #Name process convertion.
        _NAME_PROCESS = '_ILU'

        #Extraer el nombre del archivo segun la ruta asiganada.
        nameFile2Create = _FILE_PATH.split('/')
        if(_DEBUG_ON):
            print("Procces to delete / ")
            #print('OK: ', entry2.strip('hdr').strip('.').lstrip())
            print(nameFile2Create)
            print( nameFile2Create[len(nameFile2Create)-1] )
            print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
        if(not _DEBUG_ON):
            if(_ALG == "MEAN"):
                Data2Save = correctIlumNumpy( ArrayCalPoint, DataRecv[:,:,:], SpectralReference,1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print('>Save ILU MEAN')
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            elif(_ALG == "MEDIAN"):
                Data2Save = correctIlumNumpyMEDIAN( ArrayCalPoint, DataRecv[:,:,:], SpectralReference,1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print('>Save ILU MEDIAN')
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            else:
                print("No define. Posible ERROR.!!!!!!!!!!!!!")
        else:
            print("If not Error Not procces data!!! Check the _DEBUG_ON")
    else:
        print("ERROR define number argument")
elif(_MOD == "RAD_OTH"):    
    if(len(sys.argv) == 8):
        _ALG = sys.argv[2]
        _FILE_PATH_REF = sys.argv[3]
        _FILE_PATH_APL = sys.argv[4]
        _MIN = int(sys.argv[5])
        _MAX = int(sys.argv[6])
        _SAVE_PATH = sys.argv[7]
        if(_DEBUG_ON):
            print("Recv _MOD: ", _MOD)
            print("Recv _ALG: ", _ALG)
            print("Recv _FILE_PATH_REF: ", _FILE_PATH_REF)
            print("Recv _FILE_PATH_APL: ", _FILE_PATH_APL)
            print("Recv _MIN: ", _MIN)
            print("Recv _MAX: ", _MAX)
            print("Recv _SAVE_PATH: ", _SAVE_PATH)
            #Check \ on the end save Path!!!!!!
            print("Send EndSave \:", _SAVE_PATH[len(_SAVE_PATH)-1])
        
        DataRecv        = envi.open(_FILE_PATH_REF + '.hdr', _FILE_PATH_REF + '.hyspex')
        DataRecv_Apply  = envi.open(_FILE_PATH_APL + '.hdr', _FILE_PATH_APL + '.hyspex')
        print('LoadData Reference: ', _FILE_PATH_REF)
        print('LoadData Apply: ', _FILE_PATH_APL)

        ArrayCalPoint = np.zeros(( int(1),4))
        ArrayCalPoint[0, 0] = _MIN
        ArrayCalPoint[0, 1] = 0
        ArrayCalPoint[0, 2] = _MAX
        ArrayCalPoint[0, 3] = 1600

        n_rows  =   DataRecv_Apply.nrows 
        n_cols  =   DataRecv_Apply.ncols 
        n_bands =   DataRecv_Apply.nbands

        static_hdr = {
                        'lines': n_rows,
                        'samples': n_cols,
                        'bands': n_bands,
                        'header offset': 0,
                        'data type': 4,
                        'data ignore value': 2,
                        'default bands' : [55,41,12],
                        'byte order': 0,
                        'wavelength units': "nm",
                        'wavelength': [417.563436, 421.192819, 424.822201, 428.451583, 432.080965, 435.710347, 439.339730, 442.969112, 446.598494, 450.227876, 453.857258, 457.486640, 461.116023, 464.745405, 468.374787, 472.004169, 475.633551, 479.262934, 482.892316, 486.521698, 490.151080, 493.780462, 497.409844, 501.039227, 504.668609, 508.297991, 511.927373, 515.556755, 519.186138, 522.815520, 526.444902, 530.074284, 533.703666, 537.333048, 540.962431, 544.591813, 548.221195, 551.850577, 555.479959, 559.109341, 562.738724, 566.368106, 569.997488, 573.626870, 577.256252, 580.885635, 584.515017, 588.144399, 591.773781, 595.403163, 599.032545, 602.661928, 606.291310, 609.920692, 613.550074, 617.179456, 620.808839, 624.438221, 628.067603, 631.696985, 635.326367, 638.955749, 642.585132, 646.214514, 649.843896, 653.473278, 657.102660, 660.732043, 664.361425, 667.990807, 671.620189, 675.249571, 678.878953, 682.508336, 686.137718, 689.767100, 693.396482, 697.025864, 700.655247, 704.284629, 707.914011, 711.543393, 715.172775, 718.802157, 722.431540, 726.060922, 729.690304, 733.319686, 736.949068, 740.578451, 744.207833, 747.837215, 751.466597, 755.095979, 758.725361, 762.354744, 765.984126, 769.613508, 773.242890, 776.872272, 780.501655, 784.131037, 787.760419, 791.389801, 795.019183, 798.648565, 802.277948, 805.907330, 809.536712, 813.166094, 816.795476, 820.424858, 824.054241, 827.683623, 831.313005, 834.942387, 838.571769, 842.201152, 845.830534, 849.459916, 853.089298, 856.718680, 860.348062, 863.977445, 867.606827, 871.236209, 874.865591, 878.494973, 882.124356, 885.753738, 889.383120, 893.012502, 896.641884, 900.271266, 903.900649, 907.530031, 911.159413, 914.788795, 918.418177, 922.047560, 925.676942, 929.306324, 932.935706, 936.565088, 940.194470, 943.823853, 947.453235, 951.082617, 954.711999, 958.341381, 961.970764, 965.600146, 969.229528, 972.858910, 976.488292, 980.117674, 983.747057, 987.376439, 991.005821, 994.635203]
        }

        #Name process convertion.
        _NAME_PROCESS = '_RAD_OTH'

        #Extraer el nombre del archivo segun la ruta asiganada.
        nameFile2Create = _FILE_PATH_APL.split('/')
        if(_DEBUG_ON):
            print("Procces to delete / ")
            #print('OK: ', entry2.strip('hdr').strip('.').lstrip())
            print(nameFile2Create)
            print( nameFile2Create[len(nameFile2Create)-1] )
            print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
        if(not _DEBUG_ON):
            if(_ALG == "MEAN"):
                Data2Save = radiometricResponseNumpy_OtherPath(ArrayCalPoint, DataRecv[:,:,:], DataRecv_Apply[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print(">Save RAD_OTH MEAN")
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            if(_ALG == "MEDIAN"):
                Data2Save = radiometricResponseNumpyMEDIAN_OtherPath(ArrayCalPoint, DataRecv[:,:,:], DataRecv_Apply[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print(">Save RAD_OTH MEDIAN")
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            else:
                print("No define. Posible ERROR.!!!!!!!!!!!!!")   
        else:
            print("If not Error Not procces data!!! Check the _DEBUG_ON")
    else:
        print("ERROR define number argument") 
elif(_MOD == "ILU_OTH"):        
    if(len(sys.argv) == 8):
        _ALG = sys.argv[2]
        _FILE_PATH_REF = sys.argv[3]
        _FILE_PATH_APL = sys.argv[4]
        _MIN = int(sys.argv[5])
        _MAX = int(sys.argv[6])
        _SAVE_PATH = sys.argv[7]

        if(_DEBUG_ON):
            print("Recv _MOD: ", _MOD)
            print("Recv _ALG: ", _ALG)
            print("Recv _FILE_PATH_REF: ", _FILE_PATH_REF)
            print("Recv _FILE_PATH_APL: ", _FILE_PATH_APL)
            print("Recv _MIN: ", _MIN)
            print("Recv _MAX: ", _MAX)
            print("Recv _SAVE_PATH: ", _SAVE_PATH)
            #Check \ on the end save Path!!!!!!
            print("Send EndSave \:", _SAVE_PATH[len(_SAVE_PATH)-1])

        DataRecv        = envi.open(_FILE_PATH_REF + '.hdr', _FILE_PATH_REF + '.hyspex')
        DataRecv_Apply  = envi.open(_FILE_PATH_APL + '.hdr', _FILE_PATH_APL + '.hyspex')
        print('LoadData Reference: ', _FILE_PATH_REF)
        print('LoadData Apply: ', _FILE_PATH_APL)

        ArrayCalPoint = np.zeros(( int(1),4))
        ArrayCalPoint[0, 0] = _MIN
        ArrayCalPoint[0, 1] = 0
        ArrayCalPoint[0, 2] = _MAX
        ArrayCalPoint[0, 3] = 1600

        n_rows  =   DataRecv_Apply.nrows 
        n_cols  =   DataRecv_Apply.ncols 
        n_bands =   DataRecv_Apply.nbands

        static_hdr = {
                        'lines': n_rows,
                        'samples': n_cols,
                        'bands': n_bands,
                        'header offset': 0,
                        'data type': 4,
                        'data ignore value': 2,
                        'default bands' : [55,41,12],
                        'byte order': 0,
                        'wavelength units': "nm",
                        'wavelength': [417.563436, 421.192819, 424.822201, 428.451583, 432.080965, 435.710347, 439.339730, 442.969112, 446.598494, 450.227876, 453.857258, 457.486640, 461.116023, 464.745405, 468.374787, 472.004169, 475.633551, 479.262934, 482.892316, 486.521698, 490.151080, 493.780462, 497.409844, 501.039227, 504.668609, 508.297991, 511.927373, 515.556755, 519.186138, 522.815520, 526.444902, 530.074284, 533.703666, 537.333048, 540.962431, 544.591813, 548.221195, 551.850577, 555.479959, 559.109341, 562.738724, 566.368106, 569.997488, 573.626870, 577.256252, 580.885635, 584.515017, 588.144399, 591.773781, 595.403163, 599.032545, 602.661928, 606.291310, 609.920692, 613.550074, 617.179456, 620.808839, 624.438221, 628.067603, 631.696985, 635.326367, 638.955749, 642.585132, 646.214514, 649.843896, 653.473278, 657.102660, 660.732043, 664.361425, 667.990807, 671.620189, 675.249571, 678.878953, 682.508336, 686.137718, 689.767100, 693.396482, 697.025864, 700.655247, 704.284629, 707.914011, 711.543393, 715.172775, 718.802157, 722.431540, 726.060922, 729.690304, 733.319686, 736.949068, 740.578451, 744.207833, 747.837215, 751.466597, 755.095979, 758.725361, 762.354744, 765.984126, 769.613508, 773.242890, 776.872272, 780.501655, 784.131037, 787.760419, 791.389801, 795.019183, 798.648565, 802.277948, 805.907330, 809.536712, 813.166094, 816.795476, 820.424858, 824.054241, 827.683623, 831.313005, 834.942387, 838.571769, 842.201152, 845.830534, 849.459916, 853.089298, 856.718680, 860.348062, 863.977445, 867.606827, 871.236209, 874.865591, 878.494973, 882.124356, 885.753738, 889.383120, 893.012502, 896.641884, 900.271266, 903.900649, 907.530031, 911.159413, 914.788795, 918.418177, 922.047560, 925.676942, 929.306324, 932.935706, 936.565088, 940.194470, 943.823853, 947.453235, 951.082617, 954.711999, 958.341381, 961.970764, 965.600146, 969.229528, 972.858910, 976.488292, 980.117674, 983.747057, 987.376439, 991.005821, 994.635203]
        }

        #Name process convertion.
        _NAME_PROCESS = '_ILU_OTH'

        #Extraer el nombre del archivo segun la ruta asiganada.
        nameFile2Create = _FILE_PATH_APL.split('/')
        if(_DEBUG_ON):
            print("Procces to delete / ")
            #print('OK: ', entry2.strip('hdr').strip('.').lstrip())
            print(nameFile2Create)
            print( nameFile2Create[len(nameFile2Create)-1] )
            print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
        if(not _DEBUG_ON):
            if(_ALG == "MEAN"):
                Data2Save = correctIlumNumpy_OtherPath(ArrayCalPoint, DataRecv[:,:,:], DataRecv_Apply[:,:,:], SpectralReference, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print(">Save ILU_OTH MEAN")
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            if(_ALG == "MEDIAN"):
                Data2Save = correctIlumNumpyMEDIAN_OtherPath(ArrayCalPoint, DataRecv[:,:,:], DataRecv_Apply[:,:,:], SpectralReference, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print(">Save ILU_OTH MEDIAN")
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            else:
                print("No define. Posible ERROR.!!!!!!!!!!!!!")
        else:
            print("If not Error Not procces data!!! Check the _DEBUG_ON")
    else:
        print("ERROR define number argument")
elif(_MOD == "RAD+ILU_OTH"):        
    if(len(sys.argv) == 8):
        _ALG = sys.argv[2]
        _FILE_PATH_REF = sys.argv[3]
        _FILE_PATH_APL = sys.argv[4]
        _MIN = int(sys.argv[5])
        _MAX = int(sys.argv[6])
        _SAVE_PATH = sys.argv[7]
        if(_DEBUG_ON):
            print("Recv _MOD: ", _MOD)
            print("Recv _ALG: ", _ALG)
            print("Recv _FILE_PATH_REF: ", _FILE_PATH_REF)
            print("Recv _FILE_PATH_APL: ", _FILE_PATH_APL)
            print("Recv _MIN: ", _MIN)
            print("Recv _MAX: ", _MAX)
            print("Recv _SAVE_PATH: ", _SAVE_PATH)
            #Check \ on the end save Path!!!!!!
            print("Send EndSave \:", _SAVE_PATH[len(_SAVE_PATH)-1])
        
        DataRecv        = envi.open(_FILE_PATH_REF + '.hdr', _FILE_PATH_REF + '.hyspex')
        DataRecv_Apply  = envi.open(_FILE_PATH_APL + '.hdr', _FILE_PATH_APL + '.hyspex')
        print('LoadData Reference: ', _FILE_PATH_REF)
        print('LoadData Apply: ', _FILE_PATH_APL)

        ArrayCalPoint = np.zeros(( int(1),4))
        ArrayCalPoint[0, 0] = _MIN
        ArrayCalPoint[0, 1] = 0
        ArrayCalPoint[0, 2] = _MAX
        ArrayCalPoint[0, 3] = 1600

        n_rows  =   DataRecv_Apply.nrows 
        n_cols  =   DataRecv_Apply.ncols 
        n_bands =   DataRecv_Apply.nbands

        static_hdr = {
                        'lines': n_rows,
                        'samples': n_cols,
                        'bands': n_bands,
                        'header offset': 0,
                        'data type': 4,
                        'data ignore value': 2,
                        'default bands' : [55,41,12],
                        'byte order': 0,
                        'wavelength units': "nm",
                        'wavelength': [417.563436, 421.192819, 424.822201, 428.451583, 432.080965, 435.710347, 439.339730, 442.969112, 446.598494, 450.227876, 453.857258, 457.486640, 461.116023, 464.745405, 468.374787, 472.004169, 475.633551, 479.262934, 482.892316, 486.521698, 490.151080, 493.780462, 497.409844, 501.039227, 504.668609, 508.297991, 511.927373, 515.556755, 519.186138, 522.815520, 526.444902, 530.074284, 533.703666, 537.333048, 540.962431, 544.591813, 548.221195, 551.850577, 555.479959, 559.109341, 562.738724, 566.368106, 569.997488, 573.626870, 577.256252, 580.885635, 584.515017, 588.144399, 591.773781, 595.403163, 599.032545, 602.661928, 606.291310, 609.920692, 613.550074, 617.179456, 620.808839, 624.438221, 628.067603, 631.696985, 635.326367, 638.955749, 642.585132, 646.214514, 649.843896, 653.473278, 657.102660, 660.732043, 664.361425, 667.990807, 671.620189, 675.249571, 678.878953, 682.508336, 686.137718, 689.767100, 693.396482, 697.025864, 700.655247, 704.284629, 707.914011, 711.543393, 715.172775, 718.802157, 722.431540, 726.060922, 729.690304, 733.319686, 736.949068, 740.578451, 744.207833, 747.837215, 751.466597, 755.095979, 758.725361, 762.354744, 765.984126, 769.613508, 773.242890, 776.872272, 780.501655, 784.131037, 787.760419, 791.389801, 795.019183, 798.648565, 802.277948, 805.907330, 809.536712, 813.166094, 816.795476, 820.424858, 824.054241, 827.683623, 831.313005, 834.942387, 838.571769, 842.201152, 845.830534, 849.459916, 853.089298, 856.718680, 860.348062, 863.977445, 867.606827, 871.236209, 874.865591, 878.494973, 882.124356, 885.753738, 889.383120, 893.012502, 896.641884, 900.271266, 903.900649, 907.530031, 911.159413, 914.788795, 918.418177, 922.047560, 925.676942, 929.306324, 932.935706, 936.565088, 940.194470, 943.823853, 947.453235, 951.082617, 954.711999, 958.341381, 961.970764, 965.600146, 969.229528, 972.858910, 976.488292, 980.117674, 983.747057, 987.376439, 991.005821, 994.635203]
        }

        #Name process convertion.
        _NAME_PROCESS = '_RAD+ILU_OTH'

        #Extraer el nombre del archivo segun la ruta asiganada.
        nameFile2Create = _FILE_PATH_APL.split('/')
        if(_DEBUG_ON):
            print("Procces to delete / ")
            #print('OK: ', entry2.strip('hdr').strip('.').lstrip())
            print(nameFile2Create)
            print( nameFile2Create[len(nameFile2Create)-1] )
            print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
        if(not _DEBUG_ON):
            if(_ALG == "MEAN"):
                Data2Save = correctIlumNumpy_OtherPath(ArrayCalPoint, radiometricResponseNumpy( ArrayCalPoint, DataRecv[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1), radiometricResponseNumpy_OtherPath(ArrayCalPoint, DataRecv[:,:,:], DataRecv_Apply[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1), SpectralReference, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print(">Save RAD+ILU_OTH MEAN")
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            if(_ALG == "MEDIAN"):
                Data2Save = correctIlumNumpyMEDIAN_OtherPath(ArrayCalPoint, radiometricResponseNumpyMEDIAN( ArrayCalPoint, DataRecv[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1), radiometricResponseNumpyMEDIAN_OtherPath(ArrayCalPoint, DataRecv[:,:,:], DataRecv_Apply[:,:,:], BLACK_REF_IMG[:,:,:], SpectralReference, IntegrationTime, 1), SpectralReference, 1)
                envi.save_image( _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.hdr', np.float32(Data2Save), metadata = static_hdr, force=True, interleave='bil', ext='.hyspex')
                print(">Save RAD+ILU_OTH MEDIAN")
                print('SaveNamePath: ', _SAVE_PATH + str(nameFile2Create[len(nameFile2Create)-1]) + _NAME_PROCESS + '_MEDIAN_' + str(_MIN) + '_' + str(_MAX) + '.XXXXXXXXXX')
            else:
                print("No define. Posible ERROR.!!!!!!!!!!!!!")   
        else:
            print("If not Error Not procces data!!! Check the _DEBUG_ON")
    else:
        print("ERROR define number argument") 
else:
    print("Command no define")
