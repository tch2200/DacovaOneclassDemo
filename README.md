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
docker build cpp -t openvino_oneclass
docker run -it --rm -v $(pwd):/oneclass-openvino openvino_oneclass
cd cpp & mkdir build & cd build
cmake ../ -O ./
make 
./main
```

