# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:11:48 2018

@author: labelfree
"""

import MMCorePy
import PySpin
from PIL import Image
import time
import os
import shutil
import numpy as np
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import math


class blackflytest:
        
        #def __init__(self, _lcA=1+.25, _lcB=1+.25):
        def __init__(self):
            self.mmc = MMCorePy.CMMCore()
            self.mmc.loadSystemConfiguration("C:\\Program Files\\Micro-Manager-2.0beta_20180724\\ASITiger_pe4000_MLC.cfg")
            self.mmc.setProperty("LED:X:31", "State", "Open")
            self.mmc.setProperty("LED:X:31", "LED Intensity(%)", 100)
            self.mmc.setProperty("FilterSlider:S:33", "Label", "Position-4")
            self.system = PySpin.System.GetInstance()
            #lcA = 1+.25; 
            #lcB = 1+.25;
            #self.lcA = _lcA
            #self.lcB = _lcB
            self.imagenum = 12
            self.settlingTime = 0.5


        def snapB(self, lcA, lcB):
        #    swing = 0.25; #(low_B + high_B) / 2; 
        #    init_A = low_A + swing; init_B = low_B + swing;
            cam_list = self.system.GetCameras()
            cam = cam_list.GetByIndex(0)
        #    cam.EndAcquisition() ##########################################
            cam.Init()
            cam.BeginAcquisition()
            image = cam.GetNextImage()
            #image.Save( f'Python_programs/Jaehee_SpinTest/Test_Images/Test%02f/LCtest_a_%04f_b_%04f.tiff' % (self.imagenum, lcA, lcB))
            image.Save( f'Python_programs/Jaehee_SpinTest/Test_Images/Test{self.imagenum}/LCtest_a_{lcA}_b_{lcB}.tiff')
            print("saved image")
            cam.EndAcquisition() 
            #return image;
            
            
        def BlackflySP_GD(self, lcA, lcB):
            ## Initial Values
        #    swing = 0.25; #(low_B + high_B) / 2; 
        #    init_A = low_A + swing; init_B = low_B + swing;
        #    init_A = init_A.item(); init_B = init_B.item();
            self.mmc.setProperty("MeadowlarkLcOpenSource", "Retardance LC-A [in waves]", lcA ); 
            self.mmc.setProperty("MeadowlarkLcOpenSource", "Retardance LC-B [in waves]", lcB );
            time.sleep(self.settlingTime)

        def separatePol(self, lcA, lcB):
        #    folder_path = f"Python_programs/Jaehee_SpinTest/Test_Images/Test{num}"
        #    names = os.listdir(folder_path)
        #    folder_names = [names[1] for names in os.walk(folder_path)]
        #    compiled_img = []
        #    
        #    for fname in os.listdir(folder_path):
        #        img = Image.open(os.path.join(folder_path, fname))
        #        imarray = np.array(img)
        #        compiled_img.append(imarray)
        #    
        #    compiled_img = np.array(compiled_img)
            #self.path = f"Python_programs/Jaehee_SpinTest/Test_Images/Test{self.imagenum}"
            self.path = f"Python_programs/Jaehee_SpinTest/Test_Images/Test{self.imagenum}/LCtest_a_{lcA}_b_{lcB}.tiff"
            #fname = os.listdir(self.path)
            #img = Image.open( os.path.join(self.path, fname[-1] ))
            img = Image.open(self.path)
            compiled_img = np.array(img)
            
            r = 2448; c = 2048
            rr = 1224; cc = 1024
            im_0 = []; im_1 = []; im_2 = []; im_3 = [];
            
            bk = 0
            #wt = 31733.74
            
        
            im_pick_1row = []; im_pick_2row = []; 
            im = compiled_img
            im_pick_1row.append(im[0:c:2])
            im_pick_2row.append(im[1:c:2])
            ip1r = im_pick_1row[0]; 
            im_0.append(ip1r[:,0:r:2]); 
            im_1.append(ip1r[:,1:r:2]); 
            ip2r = im_pick_2row[0]; im_2.append(ip2r[:,0:r:2]); 
            im_3.append(ip2r[:,1:r:2]);
            
            im_0c = im_0[0]; im_0c = im_0c[500:600, 500:600]
            im_1c = im_1[0]; im_1c = im_1c[500:600, 500:600]
            im_2c = im_2[0]; im_2c = im_2c[500:600, 500:600]
            im_3c = im_3[0]; im_3c = im_3c[500:600, 500:600]
            
            average_int_stack0 =  im_0c[0].mean() -bk 
            average_int_stack1 =  im_1c[0].mean() -bk 
            average_int_stack2 = im_2c[0].mean() -bk 
            average_int_stack3 =  im_3c[0].mean() -bk 
        
            return average_int_stack0, average_int_stack1, average_int_stack2, average_int_stack3;
        
        def setAndSnap(self, lcA, lcB):
            self.BlackflySP_GD(lcA, lcB);
            self.snapB(lcA, lcB);
            s0, s1, s2, s3 = self.separatePol(lcA, lcB);
            diff1 = s0 - s3;
            diff2 = s1 - s2;
            s_out = math.sqrt(diff1**2 + diff2**2)
            return s_out
        
        def runOptimization(self, LC):
            if LC[0] < 1.6 and LC[1] < 1.6: 
                self.BlackflySP_GD(LC[0], LC[1]);
                self.snapB(LC[0], LC[1]);
                s0, s1, s2, s3 = self.separatePol(LC[0], LC[1]);
                diff1 = s0 - s3;
                diff2 = s1 - s2;
                s_out = math.sqrt(diff1**2 + diff2**2)
                print(s_out)
                return s_out
            else:
                LC[0] = 1.59; LC[1] = 1.59;
                self.BlackflySP_GD(LC[0], LC[1]);
                self.snapB(LC[0], LC[1]);
                s0, s1, s2, s3 = self.separatePol(LC[0], LC[1]);
                diff1 = s0 - s3;
                diff2 = s1 - s2;
                s_out = math.sqrt(diff1**2 + diff2**2)
                print(s_out)
                return s_out



    
    
    
    
    
    
    
    