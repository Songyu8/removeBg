import fire 
import glob
from tqdm import tqdm
from PIL import Image,ImageDraw
from ultralytics import YOLO
import numpy as np
import torch
from torchvision import transforms
import  os

def removeBg(**kwargs):

    device = kwargs.get('device')
    remove_img_path=kwargs.get('remove_img_path')

    model = YOLO('yolov8n-seg.pt')

    img_path=remove_img_path

    print(img_path)

    results = model(img_path,device=device,stream=False)

    data_transforms = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    for result in tqdm(results,desc='removerBg',leave=False):


        # results = model.predict(img_path,device="cuda:7")

        img = Image.open(result.path)

        if result.masks is not None and result.masks.data is not None:

            mask_alpha = result.masks.data

            print(mask_alpha.shape)

            img = data_transforms(img)

            image = img

            C, H, W = img.shape #[3,256,128]


            # image_array_copy = image_array.copy()

            # print(image_tensor.shape)

            msk = torch.nn.functional.interpolate(mask_alpha[0:1,:,:].unsqueeze(0), size=(H, W), mode='bilinear', align_corners=True)#[1,1,256,128]#mask_alpha[0:1] is people 

            msk=msk.squeeze(dim=0).to(device)

            img=img.to(device)



            segmented_img = (img * msk).cpu()


            img_np = segmented_img.numpy()  # 将张量转换为NumPy数组
            img_np = ((img_np - img_np.min()) * (255 / (img_np.max() - img_np.min()))).clip(0, 255).astype(np.uint8)  # 将数组值缩放到0到255的范围内，并转换为8位无符号整数
            result_img = Image.fromarray(img_np.transpose(1, 2, 0))  # 将数组转换为图像

            result_img.save(os.path.join(remove_img_path,os.path.basename(result.path)[:-4]+'.jpg'))


if __name__=='__main__':
    import fire
    fire.Fire(removeBg)