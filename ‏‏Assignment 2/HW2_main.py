"""
Nir Schneider 316098052
Raam Kavod    205784218
"""

import numpy as np
import cv2 as cv
import dom
import os


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# init
main_path = 'C:/Users/eszte/Desktop/Personal/BGU/5th_Year/Multimedia_Compression/HW/HW2/'
DOM = dom.DOM()


"""
///////////////////////Question 1.a//////////////////////////////
"""
slant_edge_RGB = cv.imread(main_path + "slant_edge.png")
slant_edge_YCbCr = cv.cvtColor(slant_edge_RGB, cv.COLOR_RGB2YCrCb)
slant_edge_Y = slant_edge_YCbCr[::, ::, 0]
resized_slant_edge_Y = cv.resize(slant_edge_Y, (960, 540))  # only for observation
cv.imshow('Y', resized_slant_edge_Y)
part_of_slant_edge = slant_edge_Y[2050:2180, 3050:3180]
M_rotate = cv.getRotationMatrix2D((50, 50), 6, 1)
rotate_06 = cv.warpAffine(part_of_slant_edge, M_rotate, (150, 150))
# cv.imwrite('part_of_slant_edge.png', part_of_slant_edge)
cv.imshow('100x100 pixels', rotate_06)
cv.waitKey()
"""
///////////////////////Question 1.f//////////////////////////////
"""
# reference : https://github.com/u-onder/mtf.py/blob/main/mtf.py
"""
///////////////////////Question 2.a//////////////////////////////
"""
car_motion_image = cv.imread(main_path + "car_motion.JPG")
car_motion = car_motion_image[1500:2300, 2700:4200, :]
# cv.imwrite('licence_plate_motion.jpg', car_motion)
cv.imshow('Cropped Image', car_motion)
cv.waitKey()

car_defocus_image = cv.imread(main_path + "car_defocus.JPG")
car_defocus = car_defocus_image[1550:2350, 2300:4000, :]
# cv.imwrite('licence_plate_defocus.jpg', car_defocus)
cv.imshow('Cropped Image', car_defocus)
cv.waitKey()
"""
///////////////////////Question 2.b-2.10/////////////////////////
"""

os.system('python deconvolution.py --angle 173 --d 63 --snr 23 .\licence_plate_motion.jpg')
os.system('python deconvolution.py  --d 115 --snr 25 .\licence_plate_defocus.jpg')

"""
///////////////////////Question 3.b//////////////////////////////
"""
pag_path = main_path + 'pag_sharp.jpeg'
pizza_path = main_path + 'pizza_sharp.jpeg'
suspicious_dog_path = main_path + 'suspicious_dog_sharp.jpeg'
supermarket_path = main_path + 'supermarket_blur.jpeg'
asian_woman_path = main_path + 'asian_woman.jpeg'
laundry_path = main_path + 'laundry_blur.jpeg'
paths = [pag_path, pizza_path, suspicious_dog_path, supermarket_path, asian_woman_path, laundry_path]
images_BGR = []
for path in paths:
    images_BGR.append(cv.imread(path))
images_RGB = []
for image in images_BGR:
    images_RGB.append(cv.cvtColor(image, cv.COLOR_BGR2RGB))
DOM_sharpness_scores = []
for image in images_RGB:
    DOM_sharpness_scores.append(DOM.get_sharpness(image))
print(f'Sharp Pag score is: {DOM_sharpness_scores[0]}')
print(f'Sharp Pizza score is: {DOM_sharpness_scores[1]}')
print(f'Sharp Suspicious Dog score is: {DOM_sharpness_scores[2]}')
print(f'Blur Supermarket score is: {DOM_sharpness_scores[3]}')
print(f'Blur Asian woman score is: {DOM_sharpness_scores[4]}')
print(f'Blur Laundry score is: {DOM_sharpness_scores[5]}')
