import argparse, cv2, json
from pathlib import Path
import time
import matplotlib.pyplot as plt

import numpy as np
from openvino.runtime import Core, Layout , Type
from openvino.preprocess import PrePostProcessor

from postprocessing import Tensor, predict_anomaly

class OpenVinoInfer:
    def __init__(self, model_path, device='CPU'):
        """
        Args:
            model_path (str): model .xml file
            input_size (int): with = height            
        """
        config_path = Path(model_path).with_suffix(".json")

        self.config = self.load_config(config_path=config_path)

        self.input_size = self.config['preprocessing']['shape'][0]
        self.n_channels = 3 if self.config['preprocessing']['color_mode'] == "rgb" else 1
        
        self.model = self.load_model(model_path, device)
        self.pre_process_config = self.config['preprocessing']
        self.threshold = self.config['model_config']['infer_threshold']
        
        self.min_area_ratio = self.config['model_config']['infer_min_area']
        self.layout = 'NHWC'
    
    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf8') as f:
            config = json.load(f)
        return config 

    def load_model(self, model_path, device):
        openvino_runtime_core = Core()
        model = openvino_runtime_core.read_model(model_path)

        n, h, w, c = 1, self.input_size, self.input_size, self.n_channels

        ppp = PrePostProcessor(model)

        # 1) Set input tensor information:
        # - input() provides information about a single model input
        # - reuse precision and shape from already available `input_tensor`
        # - layout of data is 'NHWC'
        ppp.input().tensor().set_element_type(Type.f32).set_layout(Layout('NHWC'))        

        # 2) Here we suppose model has 'NHWC' layout for input
        ppp.input().model().set_layout(Layout('NHWC'))

        # 3) Set output tensor information:
        # - precision of tensor is supposed to be 'f32'
        ppp.output().tensor().set_element_type(Type.f32)

        # 4) Apply preprocessing modifying the original 'model'
        model = ppp.build()

        compiled_model = openvino_runtime_core.compile_model(model, device)
        return compiled_model

    def pre_processing(self, img, imgsz=640):
        """ Preprocessing image:

        Args:
            img (_type_): _description_
            imgsz (int, optional): _description_. Defaults to 128.

        Returns:
            _type_: _description_
        """        
        ori_img = img.copy()
        color_mode = self.pre_process_config['color_mode']
        if color_mode == "grayscale":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif color_mode == 'rgb':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        
        if color_mode == "grayscale":
            img = img[:, :, np.newaxis]   # gray image HWC (C=1)
        img = img.astype(np.float32)
        img *= self.pre_process_config['rescale']
        
        img = np.expand_dims(img, 0)   # NHWC
        
        return img, ori_img

    def post_processing(self, input, output, ori_img, min_area_ratio=None, threshold=None):


        output = next(iter(output.values()))
        shape = self.pre_process_config['shape']
        
        if min_area_ratio is None:
            min_area = self.config['model_config']['infer_min_area'] * shape[0] * shape[1]
        else:
            min_area = min_area_ratio * shape[0] * shape[1]        

        if threshold is None:
            threshold =  self.config['model_config']['infer_threshold']        

        tensor_test = Tensor(
            imgs_input=input,
            imgs_pred=output,
            vmin=self.pre_process_config['vmin'],
            vmax=self.pre_process_config['vmax'],
            method=self.config['model_config']['loss'],
            dtype='float64',
            filenames="xxx",
        )
        y_pred, coco_result = predict_anomaly(
            resmaps=tensor_test.resmaps, min_area=min_area, threshold=threshold,
            original_image=ori_img, is_get_coco_result=True
        )

        coco_result.update({
                "heatmap": cv2.resize(tensor_test.resmaps[0], (int(shape[0]), int(shape[1])), interpolation=cv2.INTER_NEAREST)
            })
        if y_pred[0] == 1:
            return "bad", coco_result
        else:
            return "normal", coco_result

    def __call__(self, image):
        """
        Args:
            image (np.ndarray): cv2 image (BGR)
        """
        # input_tensor : RGB image
        input_tensor, ori_img = self.pre_processing(image, imgsz=self.input_size)
        output = self.model.infer_new_request(
            {
                0: input_tensor
            }
        )        
        result = self.post_processing(
            input=input_tensor,
            output=output,
            ori_img=ori_img,
        )
        return result

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--model', 
        help='path to openvino .xml file',
        type=str, 
        default= "model_20221222_9229_664304.xml"
    )

    parser.add_argument(
        '--image',
        help='path to image',
        type=str,
        default="./examples/inputs/demo.png"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    infer = OpenVinoInfer(
        model_path=args.model,        
        device='CPU',
    )        

    for i in range(3):
        image = cv2.imread(args.image)
        start_time = time.time()
        _class, result = infer(image)
        print("process time: {}".format(time.time() - start_time))
        print("Class pred: ", _class)        
        ### VISUALIZE RESULT ###
        segment = result['segmentation']
        bbox = result['bbox']
        heatmap = result['heatmap']        

        image_bin = np.zeros_like(image)
        contours = [np.array(seg).reshape((-1, 2)) for seg in segment]
        cv2.drawContours(image_bin, np.array(contours, dtype=object), -1, (255, 255, 255), 3)

        image_draw_box = image.copy()
        [cv2.rectangle(image_draw_box, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1) for box in bbox]

        num_cols = 4
        figure_size = (num_cols * 5, 5)
        figure, axis = plt.subplots(1, num_cols, figsize=figure_size)
        figure.subplots_adjust(right=0.9)
        axis[0].imshow(image, vmin=0, vmax=255)
        axis[0].title.set_text("original image")

        axis[1].imshow(heatmap, cmap='viridis')
        axis[1].title.set_text("Predicted Heat Map")

        axis[2].imshow(image_bin, cmap='gray', vmin=0, vmax=255)
        axis[2].title.set_text("Predicted mask")

        axis[3].imshow(image_draw_box, vmin=0, vmax=255)
        axis[3].title.set_text("Predicted bbox")

        figure.canvas.draw()
        plt.savefig("examples/output_openvino_python/demo_python.png")

        break