







# Tello-Fly-Through-Hoop
Flying through hoops with Tello drone.

## Demo Video
https://github.com/R2D2-like/Tello-Fly-Through-Hoop/assets/103891981/decf273a-676d-4ef3-89c5-ed6a8a78e9d0



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
