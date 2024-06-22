



# Tello-Fly-Through-Hoop
Flying through hoops with Tello drone.

## Demo Video
https://github.com/R2D2-like/Tello-Fly-Through-Hoop/assets/103891981/decda4b2-b5eb-4ad1-9488-0399cddd042b

## Environment
- Ubuntu 20.04 (using docker)

## Get start (Install)
```
cd Tello-Fly-Through-Hoop
make build_docker
make run_docker
(in docker) ln -s /root/external/h264decoder/build/h264decoder.cpython-38-x86_64-linux-gnu.so /root/Tello-Fly-Through-Hoop/scripts/
```

## Fly through hoops
```
python3 scripts/main.py
```
