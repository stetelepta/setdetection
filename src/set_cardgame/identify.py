## imports
import os
import sys
import numpy as np
import cv2
import logging
import imutils
from matplotlib import pyplot as plt
from pathlib import Path 
from pylab import array, plot, show, axis, arange, figure, uint8 


# setup logger
logger = logging.getLogger(__name__)


# https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
def order_points(pts):
    pts = pts.reshape((4, 2))
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect


def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped


def increase_contrast(img, f=1.8):

    assert img.dtype == 'uint8', "image should be of type 'uint8'"

    # Image data
    maxIntensity = 255.0 # we expect integers

    # Increase intensity such that dark pixels become much brighter, bright pixels become slightly bright
    newImage = maxIntensity*(img/maxIntensity)**f
    newImage = array(newImage, dtype=uint8)

    return newImage


def get_largest_contours(contours, zscore_threshold=2):
    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))

    # sort on area, largest first
    sorted_areas_contours = np.array(sorted(zip(areas, contours), key=lambda x: x[0], reverse=True))

    # return areas and contours
    sorted_areas = sorted_areas_contours[:, 0]
    sorted_contours = sorted_areas_contours[:, 1]
    
    # use z-score to identify exceptionally large contours
    cond_large = (sorted_areas - sorted_areas.mean()) / sorted_areas.std() > zscore_threshold
    
    # return largest contours
    return sorted_contours[cond_large]


def preprocess(img):
    # increase contrast
    img = increase_contrast(img)

    # convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # blur image
    blur = cv2.GaussianBlur(gray, (1, 1), 1000)

    # threshold
    flag, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # erode
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    img_erode = cv2.erode(thresh, kernel, iterations=4)
    
    return img_erode


def valid_ratio(short_side, long_side, thresh_low=0.5, thresh_high=0.8):
    if short_side / long_side < thresh_low:
        logger.debug("invalid ratio: image to long")
        return False
    if short_side / long_side > thresh_high:
        logger.debug("invalid ratio: image too square")
        return False
    return True


def identify_images(img, target_size):
    # save a copy
    img_orig = img.copy()

    # preprocess image
    img = preprocess(img)

    # find contours
    contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]

    # get largest contours
    large_contours = get_largest_contours(contours, zscore_threshold=2)
    
    if len(large_contours) == 0:
        logger.warning("no large contours found, using all contours")
        large_contours = contours
    identified_images = []
    bboxes = []
    for i, c in enumerate(large_contours):
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect).astype(int)
        warped = four_point_transform(img_orig.copy(), box)
        
        # get width and heigt of warped image
        w = warped.shape[1]
        h = warped.shape[0]

        short_side = w if w < h else h
        long_side = w if w > h else h
        
        if not valid_ratio(short_side, long_side, thresh_low=0.5, thresh_high=0.8):
            logger.debug(f"invalid ratio: s/l: {short_side / long_side}")
            continue

        # make sure image is landscape
        if h >= w:
            warped = imutils.rotate_bound(warped, 90)

        # resize image to target size
        resized = cv2.resize(warped, target_size, interpolation=cv2.INTER_AREA)
        
        # add image to output array
        identified_images.append(resized)
        bboxes.append(box)
    logger.debug(f"nr warped: {len(identified_images)}, nr large contours: {len(large_contours)}")
    
    return np.array(identified_images).astype(float), np.array(bboxes)
