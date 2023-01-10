import argparse, cv2, json, sys
import numpy as np
import logging as log
from pathlib import Path
from postprocessing import *
import onnxruntime as rt

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

class OnnxInfer:
    def __init__(self, onnx_file, device='CPU'):
        """
        Args:
            model_path (str): model .onnx file                        
        """
        config_path = Path(onnx_file).with_suffix(".json")

        self.onnx_file = onnx_file
        self.device = device.lower()
        self.config = self.load_config(config_path=config_path)

        self.input_size = self.config['preprocessing']['shape'][0]
        self.n_channels = 3 if self.config['preprocessing']['color_mode'] == "rgb" else 1
        
        self.model = self.__init_model()
        self.pre_process_config = self.config['preprocessing']
        self.threshold = self.config['model_config']['infer_threshold']        
        self.min_area_ratio = self.config['model_config']['infer_min_area']

        self.input_name = self.model.get_inputs()[0].name
    
    def load_config(self, config_path):
        with open(config_path, 'r', encoding='utf8') as f:
            config = json.load(f)
        return config 

    def __init_model(self):
        assert self.device in ["cpu", "gpu"], "{} not in allowed device list".format(
            self.device
        )
        sess_options = rt.SessionOptions()

        # Set graph optimization level
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = rt.InferenceSession(
            self.onnx_file,
            sess_options=sess_options,
            providers=[
                "CPUExecutionProvider"
                if self.device == "cpu"
                else "CUDAExecutionProvider"
            ],
        )

        return session

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

        output = output[0]
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
        output = self.model.run(None, {self.input_name: input_tensor})

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
        help='path to openvino .onnx file',
        type=str, 
        default= "weights/onnx/model_20221222_9229_664304.onnx"
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
    infer = OnnxInfer(
        onnx_file=args.model,        
        device='CPU',
    )        

    for i in range(3):
        image = cv2.imread(args.image)
        start_time = time.time()
        _class, result = infer(image)

        print("Process time: {}".format(time.time() - start_time))
        print("Class pred: ", _class)        
        ### VISUALIZE RESULT ###
        segment = result['segmentation']
        bbox = result['bbox']
        heatmap = result['heatmap']        

        image_bin = np.zeros_like(image)
        contours = [np.array(seg).reshape((-1, 2)) for seg in segment]
        cv2.drawContours(image_bin, np.array(contours), -1, (255, 255, 255), 3)

        image_draw_box = image.copy()
        [cv2.rectangle(image_draw_box, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 1) for box in bbox]

        num_cols = 4
        figure_size = (num_cols * 5, 5)
        figure, axis = plt.subplots(1, num_cols, figsize=figure_size)
        figure.subplots_adjust(right=0.9)
        axis[0].imshow(image, vmin=0, vmax=255)
        axis[0].title.set_text("Original Image")

        axis[1].imshow(heatmap, cmap='viridis')
        axis[1].title.set_text("Predicted Heat Map")

        axis[2].imshow(image_bin, cmap='gray', vmin=0, vmax=255)
        axis[2].title.set_text("Predicted Mask")

        axis[3].imshow(image_draw_box, vmin=0, vmax=255)
        axis[3].title.set_text("Predicted Bbox")

        figure.canvas.draw()
        plt.savefig("examples/output_onnx_python/demo_python.png")

        break