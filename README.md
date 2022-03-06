# pose_compare
compare video of poses to labels and return a score of similarity

# Installation
for Ubuntu

## 1-installation packages

```
pip install -r requirements.txt
```

## 2-run setup.py

```
python3 setup.py build develop
```

## 3-get models weights

link = https://drive.google.com/uc?id=1RAznEMAcXNwEWolIi6il_9BAJqREgZiY&export=download

unzip the weight model, and move weights to their respective locations

```
mkdir third_party/detector/yolo/data
mkdir third_party/detector/tracker/data

mv yolov3-spp.weights third_party/detector/yolo/data/
mv JDE-1088x608-uncertainty third_party/detector/tracker/data/
mv fast_res50_256x192.pth third_party/pretrained_models/
```