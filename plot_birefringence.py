import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt
import cv2
import bisect
import warnings
import sys
from utils.imgProcessing import nanRobustBlur,imadjust, imBitConvert, imClip
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from call_findstokes_multipleImgs import linpol_asSample
from findStokes_test import StokesImages
#from utils.imgCrop import imcrop

def plotVectorField(I, azimuth, R=20, spacing=20, clim=[None, None]): # plot vector field representation of the orientation map,
    # Currently only plot single pixel value when spacing >0. 
    # To do: Use average pixel value to reduce noise
#    retardSmooth = nanRobustBlur(retard, (spacing, spacing))
#    retardSmooth/np.nanmean(retardSmooth)
#    R = R/np.nanmean(R) 
    U, V = R*spacing*np.cos(2*azimuth), R*spacing*np.sin(2*azimuth)    
    USmooth = nanRobustBlur(U,(spacing,spacing)) # plot smoothed vector field
    VSmooth = nanRobustBlur(V,(spacing,spacing)) # plot smoothed vector field
    azimuthSmooth = 0.5*np.arctan2(VSmooth,USmooth)
    RSmooth = np.sqrt(USmooth**2+VSmooth**2)
    USmooth, VSmooth = RSmooth*np.cos(azimuthSmooth), RSmooth*np.sin(azimuthSmooth) 
   
#    azimuthSmooth  = azimuth
    nY, nX = I.shape
    Y, X = np.mgrid[0:nY, 0:nX] # notice the inversed order of X and Y        
 
#    I = histequal(I)
#    figSize = (10,10)
#    fig = plt.figure(figsize = figSize) 
    imAx = plt.imshow(I, cmap='gray', vmin=clim[0], vmax=clim[1])
    plt.title('Orientation map')                              
    plt.quiver(X[::spacing, ::spacing], Y[::spacing,::spacing], 
               USmooth[::spacing,::spacing], VSmooth[::spacing,::spacing],
               edgecolor='g',facecolor='g',units='xy', alpha=1, width=2,
               headwidth = 0, headlength = 0, headaxislength = 0,
               scale_units = 'xy',scale = 1 )  
    return imAx 

def PolColor(IAbs, retard, azimuth):
    IAbs = imBitConvert(IAbs*10**3, bit=16, norm=True) #AU, set norm to False for tiling images    
    retard = imBitConvert(retard*10**3,bit=16) # scale to pm
    azimuth = imBitConvert(azimuth/np.pi*18000,bit=16) # scale to [0, 18000], 100*degree        
    retard = cv2.convertScaleAbs(retard, alpha=0.1)
    IAbs = cv2.convertScaleAbs(IAbs, alpha=0.1)
    
    azimuth = cv2.convertScaleAbs(azimuth, alpha=0.01)
    IHsv = np.stack([azimuth, retard,IAbs],axis=2)
    IHv = np.stack([azimuth, np.ones(retard.shape).astype(np.uint8)*255,retard],axis=2)
    IHsv = cv2.cvtColor(IHsv, cv2.COLOR_HSV2RGB)    
    IHv = cv2.cvtColor(IHv, cv2.COLOR_HSV2RGB)    #
#    retardAzi = np.stack([azimuth, retard],axis=2)    
    return IHsv,IHv

##  no radian or degree conversion
def imBitConvert(im, bit=16, norm=False, limit=None):    
    im = im.astype(np.float32, copy=False) # convert to float32 without making a copy to save memory
    if norm: # local or global normalization (for tiling)
        if not limit: # if lmit is not provided, perform local normalization, otherwise global (for tiling)
            limit = [np.nanmin(im[:]), np.nanmax(im[:])] # scale each image individually based on its min and max
            
        im = (im-limit[0])/(limit[1]-limit[0])*(2**bit-1) 
    else: # only clipping, no scaling        
        im = np.clip(im, 0, 2**bit-1) # clip the values to avoid wrap-around by np.astype         
    if bit==8:        
        im = im.astype(np.uint8, copy=False) # convert to 8 bit
    else:
        im = im.astype(np.uint16, copy=False) # convert to 16 bit
    return im



# plot_biref(Sample_S0_m, SP_PolRet, SP_Slowax, ihv, ihsv, 6)
def plot_biref(IAbs, retard, azimuth, IHv, IHsv, num):
#    IAbs = imBitConvert(IAbs*10**3, bit=16, norm=True) #AU, set norm to False for tiling images    
#    retard = imBitConvert(retard*10**3,bit=16) # scale to pm
    
    #azimuth = imBitConvert(azimuth/np.pi*18000,bit=16) ## output is in degrees * 100
    
    figSize = (12,12)
    fig = plt.figure(figsize = figSize)            
                            
    plt.subplot(2,2,1)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off          
    plt.imshow(imClip(IAbs, tol=1), cmap='gray')
    plt.title('Transmission')
    plt.xticks([]),plt.yticks([])                                      
    #    plt.show()
    
    ## This plot shows Retardnace + Orientation represented in hue and value
    ax = plt.subplot(2,2,2)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off            
    imAx = plt.imshow(imadjust(IHv, bit=8)[0], cmap='hsv')  ## IHv ###################################### imAx shows the vectors  , cmap='hsv'
    plt.title('Retardance+Orientation')
    plt.xticks([]),plt.yticks([])
    divider = make_axes_locatable(ax)                             ######################################## DIVIDERS for colorbar?
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(imAx, cax=cax, orientation='vertical', ticks=np.linspace(0,255, 5))    
   # cbar.ax.set_yticklabels([r'$0^o$', r'$84^o$', r'$126^o$', r'$168^o$', r'$210^o$']) 
    cbar.ax.set_yticklabels([r'$0^o$', r'$45^o$', r'$90^o$', r'$135^o$', r'$180^o$'])  # vertically oriented colorbar                                     
    #    plt.show()
    
    ax = plt.subplot(2,2,3)    
    imAx = plotVectorField(imClip(retard,tol=1), azimuth, R=R, spacing=spacing)
    plotVectorField(retard, azimuth, R=vectorScl, spacing=spacing)   ##################################### PLOT VECTOR FIELDS
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off               
    plt.title('Retardance(nm)+Orientation')   
    plt.xticks([]),plt.yticks([]) 
    divider = make_axes_locatable(ax)                              ######################################## DIVIDERS
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cbar = fig.colorbar(imAx, cax=cax, orientation='vertical')    
                                  
    plt.subplot(2,2,4)
    plt.tick_params(labelbottom=False,labelleft=False) # labels along the bottom edge are off            
    plt.imshow(imadjust(IHsv, bit=8)[0])
    plt.title('Transmission+Retardance\n+Orientation')  
    plt.xticks([]),plt.yticks([])                                   
  #  plt.show()
    
    ## Save 4 subplots
    if zoomin:
        figName = f'Proc_Sam__Zoomin{num}.png'
    else:
        figName = f'Proc_Sam_Spacing10_{num}.png' ################################# Change name here
    
   # plt.savefig(os.path.join(ImgOutPath, figName),dpi=dpi) 
    #plt.savefig(os.path.join("2018_10_02\LinearPol_v12", figName),dpi=dpi) ################################### Save to here
    plt.savefig(os.path.join("2018_10_02\Processed_1010_Sample_v2", figName),dpi=dpi)  ##____Change folder path__
    ## Save full image tiff files
    fileName = f'Proc_Sam_{num}.tiff'
    fileName_vecfield = f'Proc_Sam_{num}_vecfield.tiff'
   # cv2.imwrite(os.path.join(ImgOutPath, fileName), IHv)
   # cv2.imwrite(os.path.join(ImgOutPath, fileName), cv2.cvtColor(IHv, cv2.COLOR_RGB2BGR))
   # cv2.imwrite(os.path.join(ImgOutPath, fileName_vecfield), cv2.cvtColor( plotVectorField(imClip(retard,tol=1), azimuth, R=R, spacing=spacing), cv2.COLOR_RGB2BGR))     


#t = StokesImages()
#names = ['000','015','030','045','060','075','090','105','120','135','150','165','180','195','210']
#names = ['0']
names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
ave_Slowax = []
im_TL = []; im_TR = []; im_BL = []; im_BR = []; p_mat = []
ave_im_TL = []; ave_im_TR = []; ave_im_BL = []; ave_im_BR = [];
Axes_mat = []
t = StokesImages()
for n in names:
    p, PolRet, s000, s045, s090, s135, Sample_S0, Sample_S1, Sample_S2, SP_Slowax, SP_PolRet, SP_Retardance = linpol_asSample(n)
    Axes_mat.append(np.mean(SP_Slowax))
    p_mat.append(PolRet)
#######    
#    im_TL.append(s000.mean()); im_TR.append(s045.mean()); im_BL.append(s090.mean()); im_BR.append(s135.mean())
#    p_mat.append(p)
    
#    im_TL, im_TR, im_BL, im_BR = t.reshapeTo_matrix(s000, s045, s090, s135)
#    ave_imTL, ave_imTR, ave_imBL, ave_imBR = t.average_val(im_TL, im_TR, im_BL, im_BR )
#######    
    ave_Slowax.append(np.mean(SP_Slowax)*180/np.pi)
    Sample_S0_mat = np.reshape(Sample_S0, (1024,1224))
    ihsv, ihv = PolColor(Sample_S0_mat, SP_PolRet, SP_Slowax+15*np.pi/180) #+15*np.pi/180
    
    spacing=10; vectorScl=1; zoomin=False; dpi=300
    tIdx = 0; zIdx = 0; posIdx = 0
    #ImgOutPath = "2018_10_02\Processed"
    #    R=retard*IAbs
    R = SP_Slowax; R = R/np.nanmean(R) #normalization
    R = vectorScl * R

    plot_biref(Sample_S0_mat, SP_PolRet, SP_Slowax+15*np.pi/180 , ihv, ihsv, n) #+15*np.pi/180
    
###### Plot varying p values
plt.axis([0, 15, 0, 1]); plt.plot(PolRet)

x = np.linspace(0,210,15); 
#Axes_mat = np.dot(Axes_mat, 180/np.pi); 
plt.plot(x, Axes_mat)

p_mat = np.reshape(p_mat, (15,np.size(names)))
######    
#x = np.linspace(0, 210, 15)
#plt.plot(x, im_BL); plt.plot(x, im_BR); plt.plot(x, im_TL); plt.plot(x, im_TR)
