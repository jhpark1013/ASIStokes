from findStokes_test import StokesImages
import os
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from sklearn.preprocessing import normalize
import math

def linpol_asSample(num):
    ## Instantiate Class
    t = StokesImages()
    ############################################
    ## BLACK LEVEL
    #bk_mat = t.refine_filepath_singleImg("2018_09_18/on sample/degree15_exp1838_a0.3316_b0.4731", "bk.tiff")
    #bk_mat = t.refine_filepath_singleImg("2018_09_18\empty\degree15_exp1838_empty_a0.3330_b0.4839", "bk.tiff")
    bk_mat = t.refine_filepath_singleImg("2018_10_02", "bk.tiff")
    bkTL_mat, bkTR_mat, bkBL_mat, bkBR_mat =  t.separate_image_barebone(bk_mat)
    bkTL_mat, bkTR_mat, bkBL_mat, bkBR_mat = t.reshapeTo_matrix(bkTL_mat, bkTR_mat, bkBL_mat, bkBR_mat)
    bk_aveTL, bk_aveTR, bk_aveBL, bk_aveBR = t.average_val(bkTL_mat, bkTR_mat, bkBL_mat, bkBR_mat)
    
    ## 15deg (multiple image files)
    #deg15, names = t.multiple_images_mat("2018_09_18/on sample/degree15_exp1838_a0.3316_b0.4731")
    #deg15, names = t.multiple_images_mat("2018_09_18\empty\degree15_exp1838_empty_a0.3330_b0.4839")
    deg15, names = t.multiple_images_mat("2018_10_02\deg15")
    im_TL = []; im_TR = []; im_BL = []; im_BR = [];
    ave_im_TL = []; ave_im_TR = []; ave_im_BL = []; ave_im_BR = [];
    for i in range(0, 15): ## bk is index 0
        iTL, iTR, iBL, iBR = t.separate_image_barebone(deg15[i])
        im_TL.append(iTL); im_TR.append(iTR); im_BL.append(iBL); im_BR.append(iBR);
    for j in range(15):
        im_TL[j], im_TR[j], im_BL[j], im_BR[j] = t.reshapeTo_matrix(im_TL[j], im_TR[j], im_BL[j], im_BR[j])
        ave_imTL, ave_imTR, ave_imBL, ave_imBR = t.average_val(im_TL[j], im_TR[j], im_BL[j], im_BR[j] )
        ave_im_TL.append(ave_imTL - bk_aveTL); ave_im_TR.append(ave_imTR - bk_aveTR); 
        ave_im_BL.append(ave_imBL - bk_aveBL); ave_im_BR.append(ave_imBR - bk_aveBR);
    #x = np.arange(0, 225, 15); plt.plot(x, ave_im_0); plt.plot(x, ave_im_1); plt.plot(x, ave_im_2); plt.plot(x, ave_im_3)
    TL = np.argmax(ave_im_TL) ; TR = np.argmax(ave_im_TR); BL = np.argmax(ave_im_BL); BR = np.argmax(ave_im_BR);
    ## test: PolState = {'TL':45, 'TR':90, 'BL':135, 'BR':180};   ## TL = im0, TR = im1, BL = im2, BR = im3 
    PolState = {'TL':TL * 15, 'TR':TR * 15, 'BL':BL * 15, 'BR':BR * 15}; PolSt_sorted = sorted(PolState, key=PolState.__getitem__);
    if PolState[PolSt_sorted[-1]] > 135: PolSt_sorted = [PolSt_sorted[-1], PolSt_sorted[0], PolSt_sorted[1], PolSt_sorted[2]]
    PolDict = {PolSt_sorted[0]:PolState[PolSt_sorted[0]], PolSt_sorted[1]:PolState[PolSt_sorted[1]], PolSt_sorted[2]:PolState[PolSt_sorted[2]], PolSt_sorted[3]:PolState[PolSt_sorted[3]]}
    ################################################
    
    ## BACKGROUND  ## on sample:  Heart_exp1838_a0.3316_b0.4731    Heart2_exp4601_a0.3316_b0.4731
    ## empty: 2018_09_18\empty\Heart_exp1838           2018_09_18\empty\Heart_exp4601
    bg_mat = t.refine_filepath_singleImg("2018_10_02\Skeletal_Muscle_lc_a_0.3250_lc_b_0.4712", "Background.tiff")
    #bg_mat = t.refine_filepath_singleImg("2018_09_18\empty\Heart_exp1838", "Background.tiff")
    #bg_mat = t.refine_filepath_singleImg("2018_09_26\skel_exp310_a0.3386_b0.4824", "Background.tiff")
    bTL, bTR, bBL, bBR = t.separate_image_barebone(bg_mat) ##barebones, takes int mat
    bgTL, bgTR, bgBL, bgBR = t.reshapeTo_matrix(bTL, bTR, bBL, bBR)
    
    ## SAMPLE#________ change this_____
    #sp_mat = t.refine_filepath_singleImg("2018_09_18\on sample\Heart_exp4601_a0.3316_b0.4731", "Sample2.tiff")
    #sp_mat = t.refine_filepath_singleImg("2018_10_02\Skeletal_Muscle_lc_a_0.3250_lc_b_0.4712", "Sample6.tiff")
   # deg = deg[1:4]
   # sp_mat = t.refine_filepath_singleImg("2018_10_02\deg15", f"{num}.tiff") 
    sp_mat = t.refine_filepath_singleImg("2018_10_02\Skeletal_Muscle_lc_a_0.3250_lc_b_0.4712", f"Sample{num}.tiff")
    #sp_mat = t.refine_filepath_singleImg("2018_09_26\skel_exp310_a0.3386_b0.4824", "Sample3.tiff")
    sTL, sTR, sBL, sBR = t.separate_image_barebone(sp_mat) ##barebones, takes int mat
    spTL, spTR, spBL, spBR = t.reshapeTo_matrix(sTL, sTR, sBL, sBR)
        
    ## INSTRUMENT MATRIX ##################################################### Define positions in the instrument matrix 0, 45, 90, 135
    ## PolDict[PolSt_sorted[0]] = 0, PolSt_sorted[0] = 'TL'
    ave_im_000 = eval("ave_im_"+PolSt_sorted[0]) - eval("bk_ave"+PolSt_sorted[0]) ;  ave_im_045 = eval("ave_im_"+PolSt_sorted[1]) - eval("bk_ave"+PolSt_sorted[1]); 
    ave_im_090 = eval("ave_im_"+PolSt_sorted[2]) - eval("bk_ave"+PolSt_sorted[2]) ;  ave_im_135 = eval("ave_im_"+PolSt_sorted[3]) - eval("bk_ave"+PolSt_sorted[3]) ;
    p, PolRet,  PseudoA = t.instrument_matrix( ave_im_000, ave_im_045, ave_im_090, ave_im_135)
    ## BACKGROUND IMAGE STOAKES ############################################### Define positions in the Background matrix 0, 45, 90, 135
    b000 = eval("b"+PolSt_sorted[0]) - eval("bk_ave"+PolSt_sorted[0]); b045 = eval("b"+PolSt_sorted[1]) - eval("bk_ave"+PolSt_sorted[1]) ; 
    b090 = eval("b"+PolSt_sorted[2]) - eval("bk_ave"+PolSt_sorted[2]); b135 = eval("b"+PolSt_sorted[3]) - eval("bk_ave"+PolSt_sorted[3]) ;
    Background_S0, Background_S1, Background_S2, I_BG = t.stokes_vector_background(PseudoA, b000, b045, b090, b135)
    ## SAMPLE IMAGE STOKES ####################################################Define positions in the Sample matrix 0, 45, 90, 135
    s000 = eval("s"+PolSt_sorted[0]) - eval("bk_ave"+PolSt_sorted[0]); s045 = eval("s"+PolSt_sorted[1]) - eval("bk_ave"+PolSt_sorted[1]); 
    s090 = eval("s"+PolSt_sorted[2]) - eval("bk_ave"+PolSt_sorted[2]); s135 = eval("s"+PolSt_sorted[3]) - eval("bk_ave"+PolSt_sorted[3]);
    Sample_S0, Sample_S1, Sample_S2, S0_unscaled, S1_unscaled, S2_unscaled = t.stokes_vector_sample(PseudoA, s000, s045, s090, s135, I_BG)
    
    ## SPECIMEN PROPERTIES __________Change this___________
    S0_corrected = (Sample_S0 - Background_S0); S1_corrected = (Sample_S1 - Background_S1) ; S2_corrected = (Sample_S2 - Background_S2);
    #SP_Slowax, SP_PolRet, SP_Retardance = t.specimen_properties_diattenuation(Sample_S0, S1_corrected, S2_corrected)
    SP_Slowax, SP_PolRet, SP_Retardance = t.specimen_properties_birefringence(Sample_S0, S1_corrected, S2_corrected)
    return p, PolRet, s000, s045, s090, s135, Sample_S0, Sample_S1, Sample_S2, SP_Slowax, SP_PolRet, SP_Retardance
    
    #### PLOT ## NO axis parameters
#    fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(8, 3), ncols=4)
#    #plt.pcolor(X, Y, f(data), cmap=cm, vmin=-4, vmax=4) example
#    ax0 = plt.subplot(221); f0 = ax0.imshow(np.reshape(S0_unscaled, (1024,1224)), cmap="gray"); ax0.set_title("Transmission"); fig.colorbar(f0)
#    ax1 = plt.subplot(222); f1 = ax1.imshow(SP_Slowax, cmap="gray"); ax1.set_title("Slow Axis"); fig.colorbar(f1)
#    ax2 = plt.subplot(223); f2 = ax2.imshow(SP_PolRet, cmap="gray"); ax2.set_title("Polarization-weighted Retardance"); fig.colorbar(f2)
#    ax3 = plt.subplot(224); f3 = ax3.imshow(SP_Retardance, cmap="gray"); ax3.set_title("Retardance"); fig.colorbar(f3)
