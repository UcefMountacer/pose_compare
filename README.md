# pose_compare
compare video of poses to labels and return a score of similarity

# Installation
for Ubuntu

## 1-installation packages

```
pip install -r requirements.txt
sudo apt-get install libyaml-dev
```

## 2-run setup.py

```
cd pose_compare/third_party
python setup.py build develop
```

## 3-get models weights

link = https://drive.google.com/uc?id=1RAznEMAcXNwEWolIi6il_9BAJqREgZiY&export=download

```
cd pose_compare/weights
gdown https://drive.google.com/uc?id=1RAznEMAcXNwEWolIi6il_9BAJqREgZiY
```

unzip the weight model, and move weights to their respective locations

```
unzip -q weights.zip
```


```
cd pose_compare
mkdir third_party/detector/yolo/data
mkdir third_party/detector/tracker/data

mv weights/yolov3-spp.weights third_party/detector/yolo/data/
mv weights/JDE-1088x608-uncertainty third_party/detector/tracker/data/
mv weights/fast_res50_256x192.pth third_party/pretrained_models/
```