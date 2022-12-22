import os, cv2
import time
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.util import img_as_ubyte
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Segmentation Parameters
# float + SSIM
THRESH_MIN_FLOAT_SSIM = 0.10
THRESH_STEP_FLOAT_SSIM = 0.002
# float + L2
THRESH_MIN_FLOAT_L2 = 0.005
THRESH_STEP_FLOAT_L2 = 0.0005
# uint8 + SSIM
THRESH_MIN_UINT8_SSIM = 20
THRESH_STEP_UINT8_SSIM = 1
# uint8 + L2 (generally uneffective combination)
THRESH_MIN_UINT8_L2 = 5
THRESH_STEP_UINT8_L2 = 1

class Tensor:
    def __init__(
        self,
        imgs_input,
        imgs_pred,
        vmin,
        vmax,
        method,
        dtype="float64",
        filenames=None,
    ):
        assert imgs_input.ndim == imgs_pred.ndim == 4
        assert dtype in ["float64", "uint8"]
        assert method in ["l2", "ssim", "mssim"]

        self.method = method
        self.dtype = dtype
        self.vmin = vmin
        self.vmax = vmax
        self.filenames = filenames
        
        if imgs_input.shape[-1] == 1:
            imgs_input = imgs_input[:, :, :, 0]
            imgs_pred = imgs_pred[:, :, :, 0]
            self.cmap = "gray"
        # if RGB
        else:
            self.cmap = None

        # compute resmaps
        self.imgs_input = imgs_input
        self.imgs_pred = imgs_pred
        self.scores, self.resmaps = calculate_resmaps(
            self.imgs_input, self.imgs_pred, method, dtype
        )
        # compute maximal threshold based on resmaps
        self.thresh_max = np.amax(self.resmaps)

        # set parameters for future segmentation of resmaps
        if dtype == "float64":
            self.vmin_resmap = 0.0
            self.vmax_resmap = 1.0
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_FLOAT_SSIM
                self.thresh_step = THRESH_STEP_FLOAT_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_FLOAT_L2
                self.thresh_step = THRESH_STEP_FLOAT_L2

        elif dtype == "uint8":
            self.vmin_resmap = 0
            self.vmax_resmap = 255
            if method in ["ssim", "mssim"]:
                self.thresh_min = THRESH_MIN_UINT8_SSIM
                self.thresh_step = THRESH_STEP_UINT8_SSIM
            elif method == "l2":
                self.thresh_min = THRESH_MIN_UINT8_L2
                self.thresh_step = THRESH_STEP_UINT8_L2    

def predict_anomaly(resmaps, min_area, threshold, original_image=None, is_get_coco_result=False):

    resmaps_th = resmaps > threshold

    tmp, areas_all = label_images(resmaps_th)
    y_pred = [is_defective(areas, min_area) for areas in areas_all]
    if is_get_coco_result:
        tmp = np.where(tmp > 0, 1, 0)[0]
        tmp = np.array(tmp, dtype=np.uint8)

        mask = cv2.resize(tmp, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        coco_labels = get_coco_result_from_mask(mask, threshold_area=min_area,threshold_width=0, threshold_height=0)[0]
        return y_pred, coco_labels    
    
    return y_pred

def is_defective(areas, min_area):
    areas = np.array(areas)
    if areas[areas >= min_area].shape[0] > 0:
        return 1
    return 0

def get_coco_result_from_mask(mask, threshold_area=1, threshold_width=1, threshold_height=1):
    labels_info = []
    # opencv 4x
    contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    bbox = []
    areas = []
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, 0.02 * peri, True)
        flatten_contour = contour.flatten().tolist()
        if len(flatten_contour) > 4:
            area = cv2.contourArea(contour)
            if area < threshold_area:
                continue
            x,y,w,h = cv2.boundingRect(contour)
            if w < threshold_width or h < threshold_height:
                continue
            xmin, xmax = np.clip([x, x+w], 0, mask.shape[1])
            ymin, ymax = np.clip([y, y+h], 0, mask.shape[0])
            bbox.append([
                int(xmin), int(ymin), int(xmax), int(ymax)
            ])
            segmentation.append(flatten_contour)
            areas.append(area)

    labels_info.append({
        "segmentation": segmentation,  # list of polygons
        "bbox": bbox,
        "area": areas
    })
    return labels_info

def get_plot_name(filename, suffix):
    filename_new, ext = os.path.splitext(filename)
    filename_new = "_".join(filename_new.split("/")) + "_" + suffix + ext
    return filename_new

def calculate_resmaps(imgs_input, imgs_pred, method, dtype="float64"):
    """
    To calculate resmaps, input tensors must be grayscale and of shape (samples x length x width).
    """
    # if RGB, transform to grayscale and reduce tensor dimension to 3
    if imgs_input.ndim == 4 and imgs_input.shape[-1] == 3:
        imgs_input_gray = tf.image.rgb_to_grayscale(imgs_input).numpy()[:, :, :, 0]
        imgs_pred_gray = tf.image.rgb_to_grayscale(imgs_pred).numpy()[:, :, :, 0]
    else:
        imgs_input_gray = imgs_input
        imgs_pred_gray = imgs_pred
    
    # calculate remaps
    if method == "l2":
        scores, resmaps = resmaps_l2(imgs_input_gray, imgs_pred_gray)
    elif method in ["ssim", "mssim"]:
        scores, resmaps = resmaps_ssim(imgs_input_gray, imgs_pred_gray)
    if dtype == "uint8":
        resmaps = img_as_ubyte(resmaps)    
    return scores, resmaps

def resmaps_ssim(imgs_input, imgs_pred):
    resmaps = np.zeros(shape=imgs_input.shape, dtype="float64")
    scores = []
    for index in range(len(imgs_input)):
        img_input = imgs_input[index]
        img_pred = imgs_pred[index]
       
        score, resmap = structural_similarity(
            img_input,
            img_pred,
            win_size=11,
            gaussian_weights=True,
            multichannel=False,
            sigma=1.5,
            full=True,
        )        
        resmaps[index] = 1 - resmap
        scores.append(score)
    resmaps = np.clip(resmaps, a_min=-1, a_max=1)
    return scores, resmaps


def resmaps_l2(imgs_input, imgs_pred):
    resmaps = (imgs_input - imgs_pred) ** 2

    scores = list(np.sqrt(np.sum(resmaps, axis=0)).flatten())
    return scores, resmaps

def label_images(images_th):    
    images_labeled = np.zeros(shape=images_th.shape)
    areas_all = []
    for i, image_th in enumerate(images_th):

        cleared = clear_border(image_th)
        # label image regions
        image_labeled = label(cleared)
        # append image
        images_labeled[i] = image_labeled

        # compute areas of anomalous regions in the current image
        regions = regionprops(image_labeled)

        if regions:
            areas = [region.area for region in regions]
            areas_all.append(areas)
        else:
            areas_all.append([0])    
    return images_labeled, areas_all