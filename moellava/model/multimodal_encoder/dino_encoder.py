# # https://github.com/tsb0601/MMVP/blob/main/LLaVA/llava/model/multimodal_encoder/dino_encoder.py

# import torch
# import torch.nn as nn
# from torchvision import transforms
# 
from transformers import BitImageProcessor, AutoModel
# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

# class DINOVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()


#         self.is_loaded = False

#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
#         patch_h = 75
#         patch_w = 50
#         feat_dim = 1536
#         self.transform = transforms.Compose([
#             transforms.GaussianBlur(9, sigma=(0.1, 2.0)),  # 高斯模糊
#             transforms.Resize((patch_h * 14, patch_w * 14)),  # 调整图像大小
#             transforms.CenterCrop((patch_h * 14, patch_w * 14)),  # 中心裁剪
#             transforms.ToTensor(),  # 转换为张量
#             transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),  # 标准化
#         ])
#         #self.load_model()

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def resize_image(self, image):
#         # transform image to (224,224)
#         return transforms.Resize((224, 224))(image)

#     def load_model(self):
#         # self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
#         # self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.image_processor = self.transform

#         # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
#         # self.vision_tower = AutoModel.from_pretrained('facebook/dinov2-base')
#         self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14') ##dinov2_vitg14

#         self.vision_tower.requires_grad_(False)

#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         # print(image_forward_outs.keys())
#         image_features = image_forward_outs["x_prenorm"]
#         if self.select_feature == 'patch':
#             image_features = image_features[:, 1:]
#         elif self.select_feature == 'cls_patch':
#             image_features = image_features
#         else:
#             raise ValueError(f'Unexpected select feature: {self.select_feature}')
#         return image_features

#     @torch.no_grad()
#     def forward(self, images):
#         # if type(images) is list:
#         #     image_features = []
#         #     for image in images:
#         #         image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
#         #         image_feature = self.feature_select(image_forward_out).to(image.dtype)
#         #         image_features.append(image_feature)
#         # else:
#             # images = self.resize_image(images)
#             # print('dino_vision_encoder_image:', images.shape)
#         image_minibatch = torch.stack([self.image_processor(image) for image in images])
#         # print("image_minibatch.shape: ", image_minibatch.shape)
#         # image_minibatch = torch.stack([self.image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images])
#         image_forward_outs = self.vision_tower.forward_features(image_minibatch.to(device=self.device, dtype=self.dtype))
#         image_features = self.feature_select(image_forward_outs)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return next(self.vision_tower.parameters()).dtype

#     @property
#     def device(self):
#         return next(self.vision_tower.parameters()).device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         #return self.config.hidden_size
#         return 1536

#     @property
#     def num_patches(self):
#         #return (self.config.image_size // self.config.patch_size) ** 2
#         return 256

# https://github.com/tsb0601/MMVP/blob/main/LLaVA/llava/model/multimodal_encoder/dino_encoder.py

import torch
import torch.nn as nn
from torchvision import transforms
import os

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()


        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        #self.load_model()

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    # def resize_image(self, image):
    #     # transform image to (224,224)
    #     return transforms.Resize((224, 224))(image)
    
    def load_model(self):
        self.image_processor = BitImageProcessor.from_pretrained('facebook/dinov2-giant')

        # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs["x_prenorm"]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        # images= torch.stack([self.image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images])
        # print('DINO: rank:', os.environ['LOCAL_RANK'], 'images:', images.device, images.dtype, 'self:', self.device, self.dtype)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(dtype=image.dtype)
                image_features.append(image_feature)
        else:
            # images = self.resize_image(images)
            image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(dtype=images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return  next(self.vision_tower.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return 1536 

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        return 256