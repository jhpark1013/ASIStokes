import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.preprocessing import normalize
import math
#from utils.imgProcessing import nanRobustBlur,imadjust, imBitConvert, imClip

class StokesImages():
    def __init__(self):
#        self.instr_mat_num = 15
#        self.region_x = 500
#        self.region_y = 600
        
        self.nameTL = 0 #PolDict['TL']
        self.nameTR = 45 #PolDict['TR']
        self.nameBL = 135 #PolDict['BL']
        self.nameBR = 90 #PolDict['BR']
        
    def refine_filepath_singleImg(self, folder_path, imgname):
        iname = os.path.join(folder_path, imgname)
        folder_path = iname.replace(os.sep, '/')
        image_ = Image.open(folder_path);
        imarray = np.array(image_)
        return imarray
    
    def refine_filepath_multiImg(folder_path):
        iname = os.path.join(folder_path, imgname)
        folder_path = iname.replace(os.sep, '/')
        return folder_path
        
    def average_val(self, vTL, vTR, vBL, vBR):
        region_x = 500; region_y = 600;
        vTL = vTL[region_x:region_y, region_x:region_y]
        vTR = vTR[region_x:region_y, region_x:region_y]
        vBL = vBL[region_x:region_y, region_x:region_y]
        vBR = vBR[region_x:region_y, region_x:region_y]
        avTL = vTL.mean(); avTR = vTR.mean(); avBL = vBL. mean(); avBR = vBR.mean()
        return avTL, avTR, avBL, avBR
    
    def separate_image_barebone(self, mat):
        r = 2448; c = 2048
        im_pick_1row = []; im_pick_2row = []; 
        im = mat
        im_pick_1row.append(im[0:c:2])
        im_pick_2row.append(im[1:c:2])
        ip1r = im_pick_1row[0]; 
        im_TL = ip1r[:,0:r:2]; 
        im_TR = ip1r[:,1:r:2]; 
        ip2r = im_pick_2row[0]; im_BL = ip2r[:,0:r:2]; 
        im_BR = ip2r[:,1:r:2];
        
        iTL = np.reshape(im_TL, (1024*1224, 1))
        iTR = np.reshape(im_TR, (1024*1224, 1))
        iBL = np.reshape(im_BL, (1024*1224, 1))
        iBR = np.reshape(im_BR, (1024*1224, 1))
        return iTL, iTR, iBL, iBR
    
    def multiple_images_mat(self, folder_path):
        names = os.listdir(folder_path)
        folder_names = [names[1] for names in os.walk(folder_path)]
        compiled_img = []
        for fname in os.listdir(folder_path):
            img = Image.open(os.path.join(folder_path, fname))
            imarray = np.array(img)
            compiled_img.append(imarray)
        compiled_img = np.array(compiled_img)
        return compiled_img, names
    
    def reshapeTo_matrix(self, iTL, iTR, iBL, iBR):
        iTL_mat = np.reshape(iTL, (1024, 1224))
        iTR_mat = np.reshape(iTR, (1024, 1224))
        iBL_mat = np.reshape(iBL, (1024, 1224))
        iBR_mat = np.reshape(iBR, (1024, 1224))
        return iTL_mat, iTR_mat, iBL_mat, iBR_mat

    def instrument_matrix( self, ave_im_000, ave_im_045, ave_im_090, ave_im_135):
        IntensityVector = np.concatenate((ave_im_000, ave_im_045, ave_im_090, ave_im_135), axis=0)
        IntV = np.reshape(IntensityVector, (4,15)); IntV_check = IntV[0,:] + IntV[2,:]; ## check.. 0+90
        Imin = np.min(IntensityVector); Imax = np.max(IntensityVector)
        Inorm = (IntensityVector-Imin)/(Imax-Imin) * 2
        IntensityV = np.reshape(Inorm, (4, 15))
        temp = np.arange(0,225,16);

        oneVec = np.ones(15); vec1 = np.cos(2*temp) ; vec2 = np.sin(2*temp);
        StokesVector = np.concatenate((oneVec, vec1, vec2))        
        Stokes = np.reshape(StokesVector, (3, 15))
             
        theorA = np.transpose(np.array([[.5, 1, 0], [.5, 0, 1], [.5, -1, 0], [.5, 0 , -1]])) # 4x3
        S_diffTrans = np.dot(theorA, IntensityV)
        PolRet = np.sqrt(S_diffTrans[1,:]**2 + S_diffTrans[2,:]**2)/S_diffTrans[0,:]
        p = np.mean(PolRet)
        Stokes_p = np.concatenate((oneVec, Stokes[1,:] * p, Stokes[2,:] * p)); Stokes_p = np.reshape(Stokes_p, (3,15))
        
        Sp = np.linalg.pinv(Stokes_p)
        A = np.dot(IntensityV , Sp)
        PseudoA = np.linalg.pinv(A)
        return p, PolRet, PseudoA

    def stokes_vector_background(self, PseudoA, i000, i045, i090, i135):
        img = np.concatenate((i000, i045, i090, i135)); img = np.reshape(img, (4, 1024*1224))
        S = np.dot(PseudoA, img)
        I_BG = S[0, :]/np.max(S[0, :])
        S0 = S[0, :]/I_BG; S1 = S[1, :]/I_BG; S2 = S[2, :]/I_BG;
        return S0, S1, S2, I_BG

    def stokes_vector_sample(self, PseudoA, i000, i045, i090, i135, I_BG):
        img = np.concatenate((i000, i045, i090, i135)); img = np.reshape(img, (4, 1024*1224))
        S = np.dot(PseudoA, img)
        S0 = S[0, :]/I_BG ; S1 = S[1, :]/I_BG; S2 = S[2, :]/I_BG; ###############divide by I_BG
        return S0, S1, S2, S[0,:], S[1,:], S[2,:]
        
    def specimen_properties_birefringence(self, S0, S1, S2):
        orientationReference = 0.5 * np.pi; analyzerLeftCircular = -1;
        Slowax =( 1/2 * np.arctan2(analyzerLeftCircular * S1,S2)+ orientationReference)  % np.pi 
        Slowax = np.reshape(Slowax, (1024, 1224))
        PolRet = np.sqrt(S1**2 + S2**2)/S0; PolRet = np.reshape(PolRet, (1024, 1224))
        PolRet_norm = (PolRet-np.min(PolRet))/(np.max(PolRet)-np.min(PolRet))
        Retardance = np.arcsin(PolRet_norm)
        Retardance = np.reshape(Retardance, (1024, 1224))
        return Slowax, PolRet, Retardance

    def specimen_properties_diattenuation(self, S0, S1, S2):
        orientationReference = 0.5 * np.pi; analyzerLeftCircular = -1;
        TransmissionAx =( 1/2 * np.arctan2(S2,S1)+ orientationReference)  % np.pi ;
        TransmissionAx = np.reshape(TransmissionAx, (1024, 1224))
        PolRet = np.sqrt(S1**2 + S2**2)/S0; PolRet = np.reshape(PolRet, (1024, 1224))
        PolRet_norm = (PolRet-np.min(PolRet))/(np.max(PolRet)-np.min(PolRet))
        Retardance = np.arcsin(PolRet_norm)
        Retardance = np.reshape(Retardance, (1024, 1224))
        return TransmissionAx, PolRet, Retardance
