"""
Nir Schneider 316098052
Raam Kavod    205784218
"""

import numpy as np
import colour as c
import matplotlib.pyplot as plt
import cv2 as cv
import skimage.metrics


def image_on_notebook(image, title):
    plt.figure(figsize=(9, 12))
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title(title, fontsize=16)
    plt.show()


def plotnoise(img, mode):
    img = cv.resize(img, dsize=(0, 0), fx=0.3, fy=0.3)
    if mode is not None:
        if mode == 'gaussian' or mode == 'speckle':
            noised_img = skimage.util.random_noise(image=img, mode=mode, seed=None, clip=True, var=0.01, mean=0)
            return noised_img

        elif mode == 'blur':
            blur = cv.GaussianBlur(img, (17, 17), cv.BORDER_DEFAULT)
            noised_img = skimage.util.random_noise(image=blur, mode='gaussian', seed=None, clip=True, var=0.01)
            return noised_img

        else:
            if mode == 's&p':
                noised_img = skimage.util.random_noise(image=img, mode=mode, amount=0.005)
            else:
                noised_img = skimage.util.random_noise(image=img, mode=mode)
            return noised_img

    else:
        noised_img = skimage.util.random_noise(image=img, mode='gaussian', seed=None, clip=True, var=0, mean=0)
        return noised_img


"""
///////////////////////Question 1.3/////////////////////////////////////
"""
original_red = (1.0, 0, 0)
original_green = (0, 1.0, 0)
original_blue = (0, 0, 1.0)
original_magenta = (1.0, 0, 1.0)
original_yellow = (1.0, 1.0, 0)
original_cyan = (0, 1.0, 1.0)
neutral_8 = (5.83895907, 5.6573631, 5.94997647)
neutral_5 = (3.67963742, 3.56457128, 3.74998228)
Original_Primaries = np.array([original_red, original_green, original_blue, original_magenta,
                               original_yellow, original_cyan, neutral_8, neutral_5])
Original_XYZ = c.sRGB_to_XYZ(Original_Primaries)
original_xy = c.XYZ_to_xy(Original_XYZ)
D65 = c.SDS_ILLUMINANTS["D65"]
c.plotting.plot_sds_in_chromaticity_diagram_CIE1931([D65], standalone=False)
x, y = original_xy[:, 0], original_xy[:, 1]
plt.plot([x[0], x[1]], [y[0], y[1]], color='k', linestyle='--', linewidth=2, label="RGB triangle")
plt.plot([x[1], x[2]], [y[1], y[2]], color='k', linestyle='--', linewidth=2)
plt.plot([x[2], x[0]], [y[2], y[0]], color='k', linestyle='--', linewidth=2)
plt.plot(original_xy[0, 0], original_xy[0, 1], 'o', color=original_red, label="RED")
plt.plot(original_xy[1, 0], original_xy[1, 1], 'o', color=original_green, label="GREEN")
plt.plot(original_xy[2, 0], original_xy[2, 1], 'o', color=original_blue, label="BLUE")
plt.legend()
plt.show()
"""
///////////////////////Question 1.6/////////////////////////////////////
# """
CC2005_org = cv.imread("C:/Users/eszte/Desktop/Personal/BGU/5th_Year/Multimedia_Compression/HW/HW1/ColorChecker2005.PNG")
image_on_notebook(CC2005_org, 'Color Checker 2005')
"""
///////////////////////Question 1.7/////////////////////////////////////
"""
CC2005_restored = cv.imread(
    "C:/Users/eszte/Desktop/Personal/BGU/5th_Year/Multimedia_Compression/HW/HW1/Colorchecker_phone.jpeg")
image_on_notebook(CC2005_restored, 'Color Checker 2005 Restored Photo')

colors_org =[]
for i in range(0, 6):
    current_color_BGR = CC2005_org[286, 68 + 100 * i]
    color_rgb = [current_color_BGR[2], current_color_BGR[1], current_color_BGR[0]]
    color_xyz = c.sRGB_to_XYZ(color_rgb)
    color_xy = c.XYZ_to_xy(color_xyz)
    colors_org.append(color_xy)

for i in range(0, 2):
    current_color_BGR = CC2005_org[385, 162 + 100 * i]
    color_rgb = [current_color_BGR[2], current_color_BGR[1], current_color_BGR[0]]
    color_xyz = c.sRGB_to_XYZ(color_rgb)
    color_xy = c.XYZ_to_xy(color_xyz)
    colors_org.append(color_xy)

colors_org =[]
for i in range(0, 6):
    current_color_BGR = CC2005_org[286, 68 + 100 * i]
    color_rgb = [current_color_BGR[2], current_color_BGR[1], current_color_BGR[0]]
    color_xyz = c.sRGB_to_XYZ(color_rgb)
    color_xy = c.XYZ_to_xy(color_xyz)
    colors_org.append(color_xy)

for i in range(0, 2):
    current_color_BGR = CC2005_org[385, 162 + 200 * i]
    color_rgb = [current_color_BGR[2], current_color_BGR[1], current_color_BGR[0]]
    color_xyz = c.sRGB_to_XYZ(color_rgb)
    color_xy = c.XYZ_to_xy(color_xyz)
    colors_org.append(color_xy)

print(f'Original colors : {colors_org}')

means_restored =[]
for i in range(0, 6):
    current_color_BGR = CC2005_restored[620:735, 100 + 245 * i: 240 + 245*i]
    color_rgb = current_color_BGR[:, :, [2, 1, 0]]
    color_xyz = c.sRGB_to_XYZ(color_rgb)
    color_xy = c.XYZ_to_xy(color_xyz)
    mean_xy = [color_xy[:, :, 0].mean(), color_xy[:, :, 1].mean()]
    means_restored.append(mean_xy)

for i in range(0, 2):
    current_color_BGR = CC2005_restored[860 : 990, 335+500*i : 485+500*i]
    color_rgb = current_color_BGR[:, :, [2, 1, 0]]
    color_xyz = c.sRGB_to_XYZ(color_rgb)
    color_xy = c.XYZ_to_xy(color_xyz)
    mean_xy = [color_xy[:, :, 0].mean(), color_xy[:, :, 1].mean()]
    means_restored.append(mean_xy)

print(f'Means of restored colors: {means_restored}')
stds = []
for i in range(0, 8):
    stds.append(np.std(means_restored[i]))
print(f'STDs : {stds}')

errors = []
for i in range(0,8):
    errors.append(np.sqrt(((colors_org[i][0]) - (means_restored[i][0]))**2 +
                          ((colors_org[i][1]) - (means_restored[i][1]))**2))
print(f'Errors : {errors}')
"""
///////////////////////Question 1.8/////////////////////////////////////
"""
c.plotting.plot_sds_in_chromaticity_diagram_CIE1931([D65], standalone=False)
x, y = original_xy[:, 0], original_xy[:, 1]
plt.plot([x[0], x[1]], [y[0], y[1]], color='k', linestyle='--', linewidth=2, label="REC.709")
plt.plot([x[1], x[2]], [y[1], y[2]], color='k', linestyle='--', linewidth=2)
plt.plot([x[2], x[0]], [y[2], y[0]], color='k', linestyle='--', linewidth=2)
plt.plot(original_xy[0, 0], original_xy[0, 1], 'o', color=original_red, label="RED")
plt.plot(original_xy[1, 0], original_xy[1, 1], 'o', color=original_green, label="GREEN")
plt.plot(original_xy[2, 0], original_xy[2, 1], 'o', color=original_blue, label="BLUE")
plt.plot(original_xy[3, 0], original_xy[3, 1], 'o', color=original_magenta, label="MAGENTA")
plt.plot(original_xy[4, 0], original_xy[4, 1], 'o', color=original_yellow, label="YELLOW")
plt.plot(original_xy[5, 0], original_xy[5, 1], 'o', color=original_cyan, label="CYAN")
plt.plot(original_xy[6, 0], original_xy[6, 1], 'o', color='#808080', markersize=6, label="NEUTRAL_8")
plt.plot(original_xy[7, 0], original_xy[7, 1], 'o', color='#D3D3D3', markersize=3, label="NEUTRAL_5")
plt.legend()
plt.show()
"""
///////////////////////Question 1.9/////////////////////////////////////
"""
c.plotting.plot_sds_in_chromaticity_diagram_CIE1931([D65], standalone=False)
x, y = original_xy[:, 0], original_xy[:, 1]
plt.plot([x[0], x[1]], [y[0], y[1]], color='k', linestyle='--', linewidth=2, label="REC.709")
plt.plot([x[1], x[2]], [y[1], y[2]], color='k', linestyle='--', linewidth=2)
plt.plot([x[2], x[0]], [y[2], y[0]], color='k', linestyle='--', linewidth=2)
plt.plot(original_xy[0, 0], original_xy[0, 1], 'o', color=original_red, label="RED")
plt.plot(original_xy[1, 0], original_xy[1, 1], 'o', color=original_green, label="GREEN")
plt.plot(original_xy[2, 0], original_xy[2, 1], 'o', color=original_blue, label="BLUE")
plt.plot(original_xy[3, 0], original_xy[3, 1], 'o', color=original_magenta, label="MAGENTA")
plt.plot(original_xy[4, 0], original_xy[4, 1], 'o', color=original_yellow, label="YELLOW")
plt.plot(original_xy[5, 0], original_xy[5, 1], 'o', color=original_cyan, label="CYAN")
plt.plot(original_xy[6, 0], original_xy[6, 1], 'o', color='#808080', markersize=6, label="NEUTRAL_8")
plt.plot(original_xy[7, 0], original_xy[7, 1], 'o', color='#D3D3D3', markersize=3, label="NEUTRAL_5")
labels = ['restored blue', 'restored green', 'restored red', 'restored yellow', 'restored magenta', 'restored cyan',
          'restored neutral_8', 'restored neutral_5']
colors_plot = [original_blue, original_green, original_red, original_yellow, original_magenta,
               original_cyan, '#808080', '#D3D3D3']
for i in range(0, 8):
    plt.plot(means_restored[i][0], means_restored[i][1], '*', color=colors_plot[i], markersize=15, label=labels[i])
plt.legend()
plt.show()

"""
///////////////////////Question 2.1 + 2.2//////////////////////////////
"""
org_image = cv.imread("C:/Users/eszte/Desktop/Personal/BGU/5th_Year/Multimedia_Compression/HW/HW1/hibiscus2.tiff")
org_image_C = org_image.copy()
YCbCr_image = cv.cvtColor(org_image, cv.COLOR_BGR2YCrCb)
Y, Cb, Cr = cv.split(YCbCr_image)
Cb_420 = Cb.copy()
Cb_420[1::2, :] = Cb_420[::2, :]
Cb_420[:, 1::2] = Cb_420[:, ::2]
Cr_420 = Cb.copy()
Cr_420[1::2, :] = Cr_420[::2, :]
Cr_420[:, 1::2] = Cr_420[:, ::2]

cv.imshow('Original Image', org_image)
cv.imshow('Y channel', Y)
cv.imshow('Cb channel 4:2:0', Cb_420)
cv.imshow('Cr channel 4:2:0', Cr_420)
cv.waitKey()
"""
///////////////////////Question 2.3/////////////////////////////////////
"""
Y_420 = Y.copy()
Y_420[1::2, :] = Y_420[::2, :]
Y_420[:, 1::2] = Y_420[:, ::2]
cv.imshow('Y channel 4:2:0', Y_420)
cv.waitKey()

PSNR_Y, PSNR_Cb, PSNR_Cr = cv.PSNR(Y, Y_420), cv.PSNR(Cb, Cb_420), cv.PSNR(Cr, Cr_420)
ssim_Y = skimage.metrics.structural_similarity(Y, Y_420, channel_axis=True)
ssim_Cb = skimage.metrics.structural_similarity(Cb, Cb_420, channel_axis=True)
ssim_Cr = skimage.metrics.structural_similarity(Cr, Cr_420, channel_axis=True)
print(f'Y channel PSNR: {PSNR_Y} , SSIM: {ssim_Y}')
print(f'Cb channel PSNR: {PSNR_Cb} , SSIM: {ssim_Cb}')
print(f'Cr channel PSNR: {PSNR_Cr}  , SSIM: {ssim_Cr}')
"""
///////////////////////Question 2.6/////////////////////////////////////
"""
original_image = plotnoise(org_image, None)
gaussian_noise = plotnoise(org_image, "gaussian")
speckle_noise = plotnoise(org_image, "speckle")
poisson_noise = plotnoise(org_image, "poisson")
snp_noise = plotnoise(org_image, "s&p")
gaussian_blur = plotnoise(org_image, "blur")

mse_gaussian = skimage.metrics.mean_squared_error(original_image, gaussian_noise)
print(f'The MSE of the gaussian noise is : {mse_gaussian}')
mse_speckle = skimage.metrics.mean_squared_error(original_image, speckle_noise)
print(f'The MSE of the speckle noise is : {mse_speckle}')
mse_poisson = skimage.metrics.mean_squared_error(original_image, speckle_noise)
print(f'The MSE of the speckle noise is : {mse_speckle}')
mse_snp = skimage.metrics.mean_squared_error(original_image, snp_noise)
print(f'The MSE of the s&p noise is : {mse_snp}')
mse_gaussian_blur = skimage.metrics.mean_squared_error(original_image, gaussian_blur)
print(f'The MSE of the gaussian blur is : {mse_gaussian_blur}')

final_img1 = cv.hconcat((original_image, gaussian_noise, speckle_noise))
final_img2 = cv.hconcat((poisson_noise, snp_noise, gaussian_blur))
final_img = cv.vconcat((final_img1, final_img2))
cv.putText(final_img, 'Original Image', (45, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 3), 3, cv.LINE_AA)
cv.putText(final_img, 'Gaussian Noise', (480, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 3), 3, cv.LINE_AA)
cv.putText(final_img, 'Speckle Noise', (950, 50), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 3), 3, cv.LINE_AA)
cv.putText(final_img, 'Poisson Noise', (45, 350), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 3), 3, cv.LINE_AA)
cv.putText(final_img, 'S&P Noise', (530, 350), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 3), 3, cv.LINE_AA)
cv.putText(final_img, 'Gaussian Blur', (950, 350), cv.FONT_HERSHEY_PLAIN, 3, (255, 0, 3), 3, cv.LINE_AA)
cv.imshow('Final', final_img)
cv.waitKey()

PSNR_GN, PSNR_S, PSNR_P, PSNR_SP, PSNR_GB = cv.PSNR(original_image, gaussian_noise),\
                                            cv.PSNR(original_image, speckle_noise),\
                                            cv.PSNR(original_image, poisson_noise),\
                                            cv.PSNR(original_image, snp_noise),\
                                            cv.PSNR(original_image, gaussian_blur)

ssim_GN = skimage.metrics.structural_similarity(original_image, gaussian_noise, channel_axis=True, win_size=3)
ssim_S = skimage.metrics.structural_similarity(original_image, speckle_noise, channel_axis=True, win_size=3)
ssim_P = skimage.metrics.structural_similarity(original_image, poisson_noise, channel_axis=True, win_size=3)
ssim_SP = skimage.metrics.structural_similarity(original_image, snp_noise, channel_axis=True, win_size=3)
ssim_GB = skimage.metrics.structural_similarity(original_image, gaussian_blur, channel_axis=True, win_size=3)

print(f'Gaussian Noise PSNR: {PSNR_GN} , SSIM: {ssim_GN}')
print(f'Speckle Noise PSNR: {PSNR_S} , SSIM: {ssim_S}')
print(f'Poisson Noise PSNR: {PSNR_P}  , SSIM: {ssim_P}')
print(f'S&P Noise PSNR: {PSNR_SP} , SSIM: {ssim_SP}')
print(f'Gaussian Blur PSNR: {PSNR_GB}  , SSIM: {ssim_GB}')

"""
///////////////////////Question 3.6/////////////////////////////////////
"""
bandwidths = [1000, 2000, 3000, 4000, 6000]
PSNRs_vp9 = [30.97, 32.52, 33.47, 34.26, 35.51]
SSIMs_vp9 = [0.854, 0.878, 0.892, 0.903,  0.92]
VMAFs_vp9 = [56.895, 68.403, 74.949, 79.572, 83.3]
PSNRs_264 = [28.9, 30.82, 32.002, 32.92, 34.36]
SSIMs_264 = [0.824, 0.855, 0.886, 0.891,  0.913]
VMAFs_264 = [44.408, 58.812, 67.414, 72.959, 81.124]
PSNRs_265 = [30.33, 32.05, 33.23, 34.36, 35.61]
SSIMs_265 = [0.844, 0.872, 0.891, 0.905,  0.924]
VMAFs_265 = [50.832, 62.434, 70.086, 75.656, 83.974]

fig, axis = plt.subplots(3)
axis[0].plot(bandwidths, PSNRs_vp9, '--o', bandwidths, PSNRs_264, '--o', bandwidths, PSNRs_265, '--o')
axis[0].legend(['vp9', '264', '265'])
axis[0].set_title('PSNRs')
axis[1].plot(bandwidths, SSIMs_vp9, '--o', bandwidths, SSIMs_264, '--o', bandwidths, SSIMs_265, '--o')
axis[1].legend(['vp9', '264', '265'])
axis[1].set_title('SSIMs')
axis[2].plot(bandwidths, VMAFs_vp9, '--o', bandwidths, VMAFs_264, '--o', bandwidths, VMAFs_265, '--o')
axis[2].legend(['vp9', '264', '265'])
axis[2].set_title('VMAFs')
plt.show()
"""
///////////////////////Question 3.8/////////////////////////////////////
"""
MOS_avg_vp9 = [2, 2.03226, 3.54839, 3.70968, 3.54839]
MOS_std_vp9 = [1, 1.0483, 0.56796, 0.73908, 1.45691]
MOS_avg_264 = [2, 2.70968, 3.32258, 3.64516, 3.83871]
MOS_std_264 = [0.7303, 0.93785, 0.59928, 0.70938, 0.68784]
MOS_avg_265 = [2.67742, 3.35484, 3.03226, 3.93548, 4.12903]
MOS_std_265 = [0.701776, 0.66073, 1.016, 0.85383, 0.92166]

fig2, axis = plt.subplots(2)
axis[0].plot(bandwidths, MOS_avg_vp9, '--o', bandwidths, MOS_avg_264, '--o', bandwidths, MOS_avg_265, '--o')
axis[0].legend(['vp9', '264', '265'])
axis[0].set_title('Averages')
axis[1].plot(bandwidths, MOS_std_vp9, '--o', bandwidths, MOS_std_264, '--o', bandwidths, MOS_std_265, '--o')
axis[1].legend(['vp9', '264', '265'])
axis[1].set_title('STDs')
plt.show()
"""
///////////////////////Question 3.9/////////////////////////////////////
"""
fig3 = plt.figure
plt.plot(MOS_avg_vp9, VMAFs_vp9, '--o', MOS_avg_264, VMAFs_264, '--o', MOS_avg_265, VMAFs_265, '--o')
plt.legend(['vp9', '264', '265'])
plt.title('MOS & VMAF')
plt.show()
