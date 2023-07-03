# # YOLOV8 to predict people and remove background
## Requirements
only in Linux

- [python 3](https://www.python.org/downloads/)
- [pytorch 1.0 + torchvision](https://pytorch.org/)
- [fire](https://github.com/google/python-fire) Automatically generating command line interfaces (CLIs)

## Install all dependences libraries like YOLO
```bash
pip install ultralytics
```
## Start predict
```bash
python removeBg.py --device='cuda:0' --remove_img_path='/home/user/image' --save_path='/home/user/image_removeBg'
