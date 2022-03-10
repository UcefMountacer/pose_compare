# pose_compare
compare video of poses to labels and return a score of similarity

# Installation
for Ubuntu

## install virtualenv an setup a virtual environment

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

## installation of packages inside virtual env

```
pip3 install torch==1.10.2+cpu torchvision==0.11.3+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip3 install -r requirements.txt
```

## test

```
python third_party/demo.py --checkpoint-path third_party/checkpoint_iter_370000.pth --images data/demo.png --cpu
```

# For deployment to server


Please refer to https://www.digitalocean.com/community/tutorials/how-to-install-nginx-on-ubuntu-18-04

to install nginx

and https://www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-gunicorn-and-nginx-on-ubuntu-18-04

to deploy


Here are the predone config file that can be used directly by modifying username

(refer to above dreployment instruction in digital ocean, the files below are just so thaat it will speed up deployment for you)

**SYSTEMD**

```
#/etc/systemd/system/pose_compare.service

[Unit]
Description=gunicorn daemon
After=network.target

[Service]
User=ecs-assist-user
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
    server_name 47.107.31.64;

    location / {
        include proxy_params;
        proxy_pass http://unix:/home/ecs-assist-user/pose_compare/pose_compare.sock;
    }

}


```