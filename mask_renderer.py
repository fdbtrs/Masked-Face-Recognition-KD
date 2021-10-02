# My Own Mask Renderer

import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt

mask_img = cv2.imread("mask_img.png", cv2.IMREAD_UNCHANGED)
mask_img = mask_img.astype(np.float32)
#mask_img = mask_img / 255
def mask(img, landmark):
    # convert mxnet image to numpy array
    # img = img.asnumpy()
    # extract landmark information

    nose = (landmark[2][0], landmark[2][1])
    mouth_left = (landmark[3][0], landmark[3][1])
    mouth_right = (landmark[4][0], landmark[4][1])
    eye_left = (landmark[0][0], landmark[0][1])
    eye_right = (landmark[1][0], landmark[1][1])

    # apply random shift of fakemask
    # rs = np.random.randint(-40,40)
    # rx = np.random.randint(-10,10)

    rs = 0
    rx = 0

    # keypoints of mask image
    src_pts = np.array([np.array([678 + rx, 464 + rs]),
                        np.array([548 + rx, 614 + rs]),
                        np.array([991 + rx, 664 + rs]),
                        np.array([1009 + rx, 64 + rs]),
                        np.array([557 + rx, 64 + rs])], dtype="float32")

    # landmark of image
    dst_pts = np.array([np.array([int(nose[0]), int(nose[1])]),
                        np.array([int(mouth_left[0]), int(mouth_left[1])]),
                        np.array([int(mouth_right[0]), int(mouth_right[1])]),
                        np.array([int(eye_right[0]), int(eye_right[1])]),
                        np.array([int(eye_left[0]), int(eye_left[1])])], dtype='float32')
    #dst_pts = np.array([

    #                    np.array([int(eye_left[0]), int(eye_left[1])]), np.array([int(eye_right[0]), int(eye_right[1])]),
    #    np.array([int(nose[0]), int(nose[1])]),
    #np.array([int(mouth_left[0]), int(mouth_left[1])]),
    #                    np.array([int(mouth_right[0]), int(mouth_right[1])])
    #                    ], dtype='float32')

    # compute perspective transformation matrix. src_pts -> dst_pts
    # The output matrix is used in next step for the transformation of the mask to an output-mask which fits to the landmark of the image
    M, _ = cv2.findHomography(src_pts, dst_pts)

    # transform the mask to a mask which fits to the image
    transformed_mask = cv2.warpPerspective(
        mask_img,
        M,
        (img.shape[1], img.shape[0]),
        None,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT)

    # overlay the image with the fitting mask
    alpha_mask = transformed_mask[:, :, 3]
    alpha_image = 1.0 - alpha_mask
    for c in range(0, 3):
        img[:, :, c] = (
                alpha_mask * transformed_mask[:, :, c]
                + alpha_image * img[:, :, c]
        )
    img= (
                alpha_mask * transformed_mask
                + alpha_image * img
        )

    # convert img to mx array
    # img = mx.nd.array(img)
    # img = img.astype('uint8')
    return img
img= cv2.imread("0_501195/0.jpg")
lm = np.array([ [30.2946, 51.6963],   # left eye
                                [65.5318, 51.5014], # right eye
                                [48.0252, 71.7366], # nose
                                [33.5493, 92.3655],
                                [62.7299, 92.2041] 
                                ], dtype=np.float32 )

center_coordinates = (int(float(lm[4][0])), int(float(lm[4][1])))

# Radius of circle
radius = 5

# Blue color in BGR
color = (0, 0, 255)

# Line thickness of 2 px
thickness = 2

# Using cv2.circle() method
# Draw a circle with blue line borders of thickness of 2 px
# right eye
img = cv2.circle(img, center_coordinates, radius, color, thickness)
img = mask(img, lm)

cv2.imshow("a", img)

cv2.waitKey()

exit()

# lm = np.array([
# [38.2946, 51.6963],
# [73.5318, 51.5014],
# [56.0252, 71.7366],
# [41.5493, 92.3655],
# [70.7299, 92.2041]
# ], dtype=np.float32)




    
    
# def paint_mask(img, landmarks):
    
    
    
# #     for 

    
# #     return nimg

#           # load mask for fakemask overlay


def image_iter(path):
    image_paths = []   
    for path, subdirs, files in os.walk(path):
        for name in files:
            image_paths.append(os.path.join(path, name))    
    return image_paths

def align_images(files):
    for idx, f in enumerate(files[0:40]):
        img = cv2.imread(f)
        path = "./test/"
        save_path = path + "_" + str(idx) + ".jpg"
        img = mask(img, lm)
        cv2.imwrite(path + "_" + str(idx) + ".jpg", img)
        print(f)

imges = image_iter("./Datasets/test/")
align_images(imges)