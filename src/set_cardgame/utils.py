import os
import numpy as np
import logging
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# setup logger
logger = logging.getLogger(__name__)


def create_folder_if_not_exists(folder_path):
    '''Creates folder if not exists

    Args: 
        folder_path (pathlib.Path instance): path to the folder that needs to be created
    '''
    try:
        os.mkdir(folder_path)
        logger.info(f'directory {folder_path} created.')
    except FileExistsError:
        logger.debug(f'directory {folder_path} already exists.')


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


def merge_sets_with_image(img, image_name, bboxes, sets, line_color=(255, 22, 84), line_width=28):
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
    # setup formatter
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    # create a logger
    logger = logging.getLogger()

    # set the loglevel
    logger.setLevel(level)

    # remove previous handlers if there are any to prevent duplicate log messages
    logger.handlers = []
    
    # setup handler and add to logger
    if logfile:
        handler = logging.FileHandler(logfile)
    else:
        handler = logging.StreamHandler()

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


def plot_images(images, suptitle="", titles=None, ncols=5, figsize=(10, 10)):
    # images: list of images
    nr_images = len(images)
    nrows = np.ceil(nr_images / ncols).astype(int)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, constrained_layout=True, gridspec_kw={'wspace':0.1, 'hspace':0.1})
    
    fig.suptitle(suptitle)
    for i, img in enumerate(images):
        if nrows > 1:
            r = i // ncols
            c = i % ncols
            ax = axs[r, c]
        else:
            ax = axs[i]
        if titles and len(titles) == nr_images:
            ax.set_title(titles[i], fontsize=14)
        if len(img.shape) == 2:
            # grayscale image
            ax.imshow(img/255, cmap='gray')
        else:
            # rgb image
            ax.imshow(img/255)
    # hide the axis for all subplots
    [ax.set_axis_off() for ax in axs.ravel()]       
    return fig


def resize_image(img, target_width, target_height=None):

    scale_percentage = target_width / img.shape[1]

    if not target_height:
        target_height = int(img.shape[0] * scale_percentage)

    # resize image
    return cv2.resize(img, (target_width, target_height), interpolation = cv2.INTER_AREA)


def crop(card_img, p=5):
    h = card_img.shape[0]
    w = card_img.shape[1]
    return card_img[0+p:h-p, 0+p:w-p, :]


def zoom(card_img, p=3):
    card_img = card_img.reshape(96, 128, 3)
    cropped = crop(card_img, p=p)
    return resize_image(cropped, target_width=card_img.shape[1], target_height=card_img.shape[0]).astype('uint8')


def read_image_with_cards(path_to_image, convert_to_rgb=True):
    img = cv2.imread(str(path_to_image))
    
    if convert_to_rgb:
        # convert image to rgb
        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)
    return img


def cluster_colours(img, n_clusters=10):
    # number of clusters
    kmeans = KMeans(n_clusters=n_clusters)

    # flatten image
    img_2d = img.reshape(img.shape[0]*img.shape[1], img.shape[2])
    
    # fit image
    kmeans = kmeans.fit(img_2d)

    # centroid values
    centroids = kmeans.cluster_centers_

    # move pixel values to nearest cluster
    distances = []
    for centroid in centroids:
        distances.append(np.linalg.norm(img_2d - centroid, axis=1))
    distances = np.array(distances).T
    closed_centroid = np.argmin(distances, axis=1)
    img_reduced = centroids[closed_centroid]
    return img_reduced.reshape(96,128,3), centroids


def plot_scatter(img, img_label="", suptitle=""):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    fig.suptitle(suptitle, fontsize=14)
    for i, colors in enumerate([("red", "green"), ("red", "blue"), ("blue", "green")]):
        color_channels = {'red': 0, 'green': 1, 'blue': 2}
        channels = (color_channels[colors[0]], color_channels[colors[1]])
        x = img[:,:,channels[0]].reshape(img.shape[0]*img.shape[1])
        y = img[:,:,channels[1]].reshape(img.shape[0]*img.shape[1])

        axs[0].set_title(img_label, fontsize=12)
        axs[0].imshow(img/255)
        axs[0].axis('off')
        axs[i+1].set_title(f"{colors[0]} vs {colors[1]}")
        axs[i+1].set_xlim(0, 255)
        axs[i+1].set_ylim(0, 255)    
        axs[i+1].set_xlabel(colors[0])
        axs[i+1].set_ylabel(colors[1])
        axs[i+1].plot(x, y, linestyle="None", marker=".", color="k", markersize=2)


def plot_scatter_centroids(img, img_label="", suptitle="", centroids=[]):
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))
    fig.suptitle(suptitle, fontsize=14)
    for i, colors in enumerate([("red", "green"), ("red", "blue"), ("blue", "green")]):
        color_channels = {'red': 0, 'green': 1, 'blue': 2}
        channels = (color_channels[colors[0]], color_channels[colors[1]])
        x = img[:,:,channels[0]].reshape(img.shape[0]*img.shape[1])
        y = img[:,:,channels[1]].reshape(img.shape[0]*img.shape[1])

        axs[0].set_title(img_label, fontsize=12)
        axs[0].imshow(img/255)        
        axs[i+1].set_title(f"{colors[0]} vs {colors[1]}")
        axs[i+1].set_xlim(0, 255)
        axs[i+1].set_ylim(0, 255)    
        axs[i+1].set_xlabel(colors[0])
        axs[i+1].set_ylabel(colors[1])
        axs[i+1].grid(True)
        axs[i+1].plot(x, y, linestyle="None", marker=".", color="k", alpha=0.3, markersize=10)
        
    for centroid in centroids:
        axs[1].plot(centroid[0], centroid[1], marker=".", color="red", markersize="20", alpha=1)
        axs[2].plot(centroid[0], centroid[2], marker=".", color="red", markersize="20", alpha=1)
        axs[3].plot(centroid[2], centroid[1], marker=".", color="red", markersize="20", alpha=1)

