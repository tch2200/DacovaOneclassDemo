# Demo DacovaOneclass

## 1. Install libs
```
pip install -r requirements.txt
```
## 2. Run infer openvino python
```
python infer_openvino.py
```
## 3. Run infer onnx python
```
python infer_onnx.py
```
## 4. Run infer openvino-cpp
```
docker build openvino_cpp -t openvino_oneclass
docker run -it --rm -v $(pwd):/oneclass-openvino openvino_oneclass
cd openvino_cpp & mkdir build & cd build
cmake ../ -O ./
make 
./main
```
## 5. Run infer onnx-cpp
```
docker build onnx_cpp -t onnx_oneclass
docker run -it --rm -v $(pwd):/oneclass-onnx onnx_oneclass
cd onnx_cpp & mkdir build & cd build
cmake ../ -O ./
make 
./main
```

