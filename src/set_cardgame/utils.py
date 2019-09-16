import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt


def to_rgb3(im):
    # we can use dstack and an array copy
    # this has to be slow, we create an array with
    # 3x the data we need and truncate afterwards
    return np.asarray(np.dstack((im, im, im)), dtype=np.uint8)


def get_mask(img, bboxes):
    # draw filled contours
    img_bg = cv2.drawContours(img, bboxes, -1, (255, 255, 255), -1)

    # convert to grayscale
    img_bg = cv2.cvtColor(img_bg, cv2.COLOR_BGR2GRAY)

    # apply threshold
    flag, thresh = cv2.threshold(img_bg, 250, 255, cv2.THRESH_BINARY)

    return thresh


def merge_sets_with_image(img, image_name, bboxes, sets, line_color=(255, 22, 84), line_width=20):
    set_images = []
    
    #img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    
    for i, s in enumerate(sets, start=1):
        # mask for alpha blending
        img_mask = get_mask(img.copy(), bboxes[s])

        # foreground with contours
        img_contours = cv2.drawContours(img.copy(), bboxes[s], -1, line_color, line_width)
        
        # dark version of contours
        foreground = img_contours.copy().astype(float) / 255
        background = 0.5*(img_contours.copy() / 255)
                
        alpha = to_rgb3(img_mask).astype(float)/255
        
        # Multiply the foreground with the alpha
        foreground = cv2.multiply(alpha, foreground)

        # Multiply the background with ( 1 - alpha )
        background = cv2.multiply(1.0 - alpha, background)
        
        # Add the masked foreground and background.
        outImage = cv2.add(foreground, background)
        
        outImage = (outImage*255).astype(int)
        
        set_images.append(outImage)
    return set_images


def conditionally_decorate(decorator, condition):
    '''return decorator that applies a passed decorator if some condition is True

    See https://stackoverflow.com/questions/3773555/python3-decorating-conditionally
    '''
    def resdec(f):
        if not condition:
            return f
        return decorator(f)
    return resdec



def setup_logger(level=logging.DEBUG, logfile=None):
    assert logfile is not None, "specify a logile"

    # setup formatter
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # create a logger
    logger = logging.getLogger()

    # set the loglevel
    logger.setLevel(level)

    # remove previous handlers if there are any to prevent duplicate log messages
    logger.handlers = []
    
    # setup handler and add to logger
    handler = logging.FileHandler(logfile)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=6)
    plt.yticks(tick_marks, classes, fontsize=6)

    plt.tight_layout()

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

