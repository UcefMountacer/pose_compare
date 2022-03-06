# pose_compare
compare video of poses to labels and return a score of similarity

# Installation
for Ubuntu

## 0- install virtualenv an setup a virtual environment

1- Update the server and install python dependencies

```
sudo apt update
```

2- Install Virtual Environment

```
sudo apt-get install -y python3-venv
```

3- Navigate inside the project directory

```
cd ${HOME}/pose_compare
```

4- Create the virtual environment

```
python3 -m venv .env
```

5- Activate virtual environment and install requirements

```
source .env/bin/activate
```

## 1-installation of packages inside virtual env

```
pip install -r requirements.txt
sudo apt-get install libyaml-dev
```

## 2-run setup.py

```
cd ${HOME}/pose_compare/third_party
python setup.py build develop
```

## 3-get models weights

link = https://drive.google.com/uc?id=1RAznEMAcXNwEWolIi6il_9BAJqREgZiY&export=download

```
cd ${HOME}/pose_compare/weights
gdown https://drive.google.com/uc?id=1RAznEMAcXNwEWolIi6il_9BAJqREgZiY
```

unzip the weight model, and move weights to their respective locations

```
unzip -q weights.zip
```


```
cd ${HOME}/pose_compare

mkdir third_party/detector/yolo/data
mkdir third_party/detector/tracker/data

mv weights/yolov3-spp.weights third_party/detector/yolo/data/
mv weights/JDE-1088x608-uncertainty third_party/detector/tracker/data/
mv weights/fast_res50_256x192.pth third_party/pretrained_models/
```

# For deployment to server

Please refer to https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04

Here are the predone config file that can be used directly by modifying username

(refer to above dreployment instruction in digital ocean, the files below are just so thaat it will speed up deployment for you)

**SYSTEMD**

```
#/etc/systemd/system/pose_compare.service

[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=ubuntu
Group=www-data
Environment="APP_ENV=PRD"
WorkingDirectory=/home/ubuntu/pose_compare
ExecStart=/home/ubuntu/pose_compare/.env/bin/gunicorn --access-logfile /home/ubuntu/logs/gunicorn-website-access.log --error-logfile /home/ubuntu/logs/gunicorn-website-error.log   --workers 6 --bind unix:/home/ubuntu/pose_compare/pose_compare.sock  -m 007 wsgi:application --timeout 20


[Install]
WantedBy=multi-user.target

```

**NGINX**

(You need to change the server IP)

```
#/etc/nginx/sites-available/pose_compare


server {
    listen 80;
    server_name 18.183.56.10;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ubuntu/pose_compare/pose_compare.sock;
    }
}

```