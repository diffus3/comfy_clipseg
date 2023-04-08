import torch
import cv2
import os
import numpy as np


from PIL import Image
from comfy_extras.clipseg.models.clipseg import CLIPDensePredT
from torchvision import transforms


PATH = os.path.dirname(os.path.realpath(__file__))

PATH_WEIGHTS = os.path.join(PATH, "weights")
#PATH_EXTRAS = os.path.join('.', 'comfy_extras', 'clipseg')
#PATH_WEIGHTS = os.path.join(PATH_EXTRAS, "weights")

class ClipSeg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "image": ("IMAGE", ),
                "clip": ("STRING", {"default": "hand, foot, face"}),
                "device": (sorted(['cpu', 'cuda', 'mps', 'xpu']), { "default": 'cuda' } ),
                "width": ("INT", {"default": 352, "min": 0, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 352, "min": 0, "max": 2048, "step": 8}),
                "threshold": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1} ),
                "mode": (sorted(['average', 'sum']), { "default": 'sum' } ),
            },
        }
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "evaluate"

    CATEGORY = "image"

    def evaluate(self, image, clip, device, width, height, threshold, mode):

        print(os.path.dirname(os.path.realpath(__file__)))

        #Load the weights with the prettier output
        model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64, complex_trans_conv=True)
        model.eval()
        #model.load_state_dict(torch.load(os.path.join(PATH_WEIGHTS, 'rd64-uni-refined.pth'), map_location=torch.device('cuda')), strict=False)
        model.load_state_dict(torch.load(os.path.join(PATH_WEIGHTS, 'rd64-uni-refined.pth'), map_location=torch.device(device)), strict=False)

        print("initial image shape:")
        print(image.shape)

        #Change shape
        #[1, 512, 512, 3] -> [1, 3, 512, 512]
        image = torch.permute(image, (0, 3, 1, 2))


        print("changed shape:")
        print(image.shape)

        transform = transforms.Compose([
            #transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.Resize((height, width), antialias=True),
        ])
        transformed_image = transform(image)#.unsqueeze(0)

        print("transformed image shape:")
        print(transformed_image.shape)


        #with torch.no_grad():
        #    predictions = model(img.repeat(len(prompts),1,1,1), prompts)[0]

        prompts = clip.split(',')
        summed = torch.zeros([1, height, width])
        for prompt in prompts:
            print(prompt)
            pred = torch.zeros([1, height, width])
            with torch.no_grad():
                pred = model(transformed_image, prompt)[0]
            print(pred)
            print(pred.shape)

            summed = summed.add(torch.sigmoid(pred))
            #pred = model()
        averaged = summed.divide(len(prompts)).clamp(min=0, max=1).squeeze(0)
        summed = summed.clamp(min=0, max=1).squeeze(0)

        if (threshold >= 0):
            averaged = averaged.subtract(threshold/255).clamp(min=0, max=1).ceil()
            summed = summed.subtract(threshold/255).clamp(min=0, max=1).ceil()

        #mask = np.zeros((height, width), dtype=np.float32)
        #mask[:,:] = alpha/255

        print("averaged")
        print(averaged.shape)

        print("summed")
        print(summed.shape)

        mask = np.zeros((height, width), dtype=np.float32)
        mask = torch.from_numpy(mask)[None,]

        image_out = np.zeros((height, width, 3), dtype=np.float32)
        image_out = torch.from_numpy(image_out)[None,]

        print("mask target shape")
        print(mask.shape)

        print("image target shape")
        print(image_out.shape)

        if (mode == "average"):
            mask = averaged
        else:
            mask = summed

        print(mask)

        print("mask shape:")
        print(mask.shape)

        print("image_out shape prior to change:")
        print(image_out.shape)

        #image_out = torch.cat([mask.movedim(0,-1)]*3)[None,]
        image_out = torch.cat([mask]*3).movedim(0,-1)[None,]

        #image_out = mask.repeat(1, 1, 1, 3)
        #image_out = torch.cat((mask, image_out), 3)

        print("image_out shape after change:")
        print(image_out.shape)
        #image_out = torch.cat([mask.movedim(0,-1)]*3)[None,]
        #mask = torch.from_numpy(mask)[None,]

        return (image_out, mask,)


NODE_CLASS_MAPPINGS = {
    "ClipSeg": ClipSeg,
}