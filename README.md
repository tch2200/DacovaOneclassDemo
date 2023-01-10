# Demo DacovaOneclass

## 1. Install libs + download weight
```
pip install -r requirements.txt
```
## 2. Run infer openvino-python
```
sh scripts/infer.sh
```

## 3. Run infer openvino-cpp
```
docker build openvino_cpp -t openvino_oneclass
docker run -it --rm -v $(pwd):/oneclass-openvino openvino_oneclass
cd openvino_cpp & mkdir build & cd build
cmake ../ -O ./
make 
./main
```
## 4. Run infer onnx-cpp
```
docker build onnx_cpp -t onnx_oneclass
docker run -it --rm -v $(pwd):/oneclass-onnx onnx_oneclass
cd onnx_cpp & mkdir build & cd build
cmake ../ -O ./
make 
./main
```

