import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.preprocessing import normalize

class StokesImages:
    def __init__(self):
        self.background_path = "2018_09_18/on sample/Heart2_exp4601_a0.3316_b0.4731/Background.tiff"
        self.instr_mat_num = 15
        
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
        
    def average_val(self, v0, v1, v2, v3):
        region_x = 500; region_y = 600;
        v0 = v0[region_x:region_y, region_x:region_y]
        v1 = v1[region_x:region_y, region_x:region_y]
        v2 = v2[region_x:region_y, region_x:region_y]
        v3 = v3[region_x:region_y, region_x:region_y]
        av0 = v0.mean(); av1 = v1.mean(); av2 = v2. mean(); av3 = v3.mean()
        return av0, av1, av2, av3
    
    def separate_image_barebone(self, mat):
        r = 2448; c = 2048
        im_pick_1row = []; im_pick_2row = []; 
        im = mat
        im_pick_1row.append(im[0:c:2])
        im_pick_2row.append(im[1:c:2])
        ip1r = im_pick_1row[0]; 
        im_0 = ip1r[:,0:r:2]; 
        im_1 = ip1r[:,1:r:2]; 
        ip2r = im_pick_2row[0]; im_2 = ip2r[:,0:r:2]; 
        im_3 = ip2r[:,1:r:2];
        
        i0 = np.reshape(im_0, (1024*1224, 1))
        i1 = np.reshape(im_1, (1024*1224, 1))
        i2 = np.reshape(im_2, (1024*1224, 1))
        i3 = np.reshape(im_3, (1024*1224, 1))
        return i0, i1, i2, i3
    
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
    
    def reshapeTo_matrix(self, i0, i1, i2, i3):
        i0_mat = np.reshape(i0, (1024, 1224))
        i1_mat = np.reshape(i1, (1024, 1224))
        i2_mat = np.reshape(i2, (1024, 1224))
        i3_mat = np.reshape(i3, (1024, 1224))
        return i0_mat, i1_mat, i2_mat, i3_mat

    def instrument_matrix( self, ave_im_0, ave_im_1, ave_im_2, ave_im_3):
        IntensityVector = np.concatenate((ave_im_0, ave_im_1, ave_im_2, ave_im_3), axis=0)
        IntV = np.reshape(IntensityVector, (4,15)); IntV_check = IntV[0,:] + IntV[2,:]; ## how do i check.. 0+90
        Imin = np.amin(IntensityVector); Imax = np.amax(IntensityVector)
        Inorm = (IntensityVector-Imin)/(Imax-Imin)
        IntensityV = np.reshape(Inorm, (4, 15))
        temp = np.arange(0,225,16);
        oneVec = np.ones(15); vec1 = np.cos(2*temp) ; vec2 = np.sin(2*temp);
        StokesVector = np.concatenate((oneVec, vec1, vec2))
        Stokes = np.reshape(StokesVector, (3, 15))
        Sp = np.linalg.pinv(Stokes)
        A = np.dot(IntensityV , Sp)
        PseudoA = np.linalg.pinv(A)
        return PseudoA

    def stokes_vector_background(self, PseudoA, i0, i1, i2, i3):
        img = np.concatenate((i0, i1, i2, i3)); img = np.reshape(img, (4, 1024*1224))
        S = np.dot(PseudoA, img)
        S0 = S[0, :]; S1 = S[1, :]; S2 = S[2, :];
        I_BG = S0/np.max(S0)
       # I_BG = 1 #/(np.amax(S0) - np.amin(S0))
        return S0, S1, S2, I_BG

    def stokes_vector_sample(self, PseudoA, i0, i1, i2, i3, I_BG):
        img = np.concatenate((i0, i1, i2, i3)); img = np.reshape(img, (4, 1024*1224))
        S = np.dot(PseudoA, img)
        S0 = S[0, :]/I_BG ; S1 = S[1, :]/I_BG; S2 = S[2, :]/I_BG;
        S0 = S0/I_BG ; S1 = S1/I_BG; S2 = S2/I_BG;
        return S0, S1, S2
        
    def specimen_properties(self, S0, S1, S2):
        Slowax = 1/2 * np.arctan2(-S1,S2); Slowax = np.reshape(Slowax, (1024, 1224))
        PolRet = np.sqrt(S1**2 + S2**2)/S0; PolRet = np.reshape(PolRet, (1024, 1224))
        Retardance = np.arcsin(np.sqrt(S1**2 + S2**2)/S0); Retardance = np.reshape(Retardance, (1024, 1224))
        return Slowax, PolRet, Retardance

