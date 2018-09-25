## Instantiate Class
t = StokesImages()
## BACKGROUND  ## Heart_exp1838_a0.3316_b0.4731
bg_mat = t.refine_filepath_singleImg("2018_09_18\on sample\Heart2_exp4601_a0.3316_b0.4731", "Background.tiff")
b0, b1, b2, b3 = t.separate_image_barebone(bg_mat) ##barebones, takes int mat
bg0, bg1, bg2, bg3 = t.reshapeTo_matrix(b0, b1, b2, b3)

## SAMPLE
sp_mat = t.refine_filepath_singleImg("2018_09_18\on sample\Heart2_exp4601_a0.3316_b0.4731", "Sample0.tiff")
s0, s1, s2, s3 = t.separate_image_barebone(sp_mat) ##barebones, takes int mat
sp0, sp1, sp2, sp3 = t.reshapeTo_matrix(b0, b1, b2, b3)

## BLACK LEVEL
bk_mat = t.refine_filepath_singleImg("2018_09_18/on sample/degree15_exp1838_a0.3316_b0.4731", "bk.tiff")
bk0_mat, bk1_mat, bk2_mat, bk3_mat =  t.separate_image_barebone(bk_mat)
bk0_mat, bk1_mat, bk2_mat, bk3_mat = t.reshapeTo_matrix(bk0_mat, bk1_mat, bk2_mat, bk3_mat)
bk_ave0, bk_ave1, bk_ave2, bk_ave3 = t.average_val(bk0_mat, bk1_mat, bk2_mat, bk3_mat)

## 15deg (multiple image files)
deg15, names = t.multiple_images_mat("2018_09_18/on sample/degree15_exp1838_a0.3316_b0.4731")
im_0 = []; im_1 = []; im_2 = []; im_3 = [];
ave_im_0 = []; ave_im_1 = []; ave_im_2 = []; ave_im_3 = [];
for i in range(1, 16): ## bk is index 0
    i0, i1, i2, i3 = t.separate_image_barebone(deg15[i])
    im_0.append(i0); im_1.append(i1); im_2.append(i2); im_3.append(i3);
for j in range(15):
    im_0[j], im_1[j], im_2[j], im_3[j] = t.reshapeTo_matrix(im_0[j], im_1[j], im_2[j], im_3[j])
    ave_im0, ave_im1, ave_im2, ave_im3 = t.average_val(im_0[j], im_1[j], im_2[j], im_3[j])
    ave_im_0.append(ave_im0); ave_im_1.append(ave_im1); ave_im_2.append(ave_im2); ave_im_3.append(ave_im3);
    
## INSTRUMENT MATRIX
PseudoA = t.instrument_matrix( ave_im_0, ave_im_1, ave_im_2, ave_im_3)
## BACKGROUND IMAGE STOAKES
Background_S0, Background_S1, Background_S2, I_BG = t.stokes_vector_background(PseudoA, b0, b1, b2, b3)
## SAMPLE IMAGE STOKES
Sample_S0, Sample_S1, Sample_S2 = t.stokes_vector_sample(PseudoA, s0, s1, s2, s3, I_BG)

## SPECIMEN PROPERTIES
S1_corrected = Sample_S1 - Background_S1 ; S2_corrected = Sample_S2 - Background_S2;
SP_Slowax, SP_PolRet, SP_Retardance = t.specimen_properties(Sample_S0, S1_corrected, S2_corrected)


## PLOT
fig, (ax0, ax1, ax2, ax3) = plt.subplots(figsize=(8, 3), ncols=4)
ax0 = plt.subplot(221); f0 = ax0.imshow(np.reshape(Sample_S0, (1024,1224)), cmap="gray"); ax0.set_title("Transmission"); fig.colorbar(f0)
ax1 = plt.subplot(222); f1 = ax1.imshow(SP_Slowax, cmap="gray"); ax1.set_title("Slow Axis"); fig.colorbar(f1)
ax2 = plt.subplot(223); f2 = ax2.imshow(SP_PolRet, cmap="gray"); ax2.set_title("Polarization-weighted Retardance"); fig.colorbar(f2)
ax3 = plt.subplot(224); f3 = ax3.imshow(SP_Retardance, cmap="gray"); ax3.set_title("Retardance"); fig.colorbar(f3)




