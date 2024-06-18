# import torch
# import torch.nn as nn
# from torchvision import transforms


# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
# from transformers import LayoutLMv3Model, LayoutLMv3ImageProcessor

# class OCRVisionTower(nn.Module):
#     def __init__(self, vision_tower, args, delay_load=False):
#         super().__init__()


#         self.is_loaded = False

#         self.vision_tower_name = vision_tower
#         self.select_layer = args.mm_vision_select_layer
#         self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

#         #self.load_model()

#         if not delay_load:
#             self.load_model()
#         else:
#             self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

#     def resize_image(self, image):
#         # transform image to (224,224)
#         return transforms.Resize((224, 224))(image)

#     def load_model(self):
#         # self.image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-large")

#         # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
#         self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
#         self.vision_tower = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-large")
#         self.vision_tower.requires_grad_(False)

#         self.is_loaded = True

#     def feature_select(self, image_forward_outs):
#         image_features = image_forward_outs.hidden_states[self.select_layer]
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
#             # print('ocr_vision_encoder_image:', images.shape)
#         image_minibatch = torch.stack([self.image_processor.preprocess(i, return_tensors='pt')['pixel_values'][0] for i in images])
#         image_minibatch = self.resize_image(image_minibatch)
#         image_forward_outs = self.vision_tower(pixel_values=image_minibatch.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
#         image_features = self.feature_select(image_forward_outs)

#         return image_features

#     @property
#     def dummy_feature(self):
#         return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

#     @property
#     def dtype(self):
#         return self.vision_tower.dtype

#     @property
#     def device(self):
#         return self.vision_tower.device

#     @property
#     def config(self):
#         if self.is_loaded:
#             return self.vision_tower.config
#         else:
#             return self.cfg_only

#     @property
#     def hidden_size(self):
#         #return self.config.hidden_size
#         return 1024

#     @property
#     def num_patches(self):
#         #return (self.config.image_size // self.config.patch_size) ** 2
#         return 196
import torch
import torch.nn as nn
from torchvision import transforms
import os


from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from transformers import LayoutLMv3Model, LayoutLMv3ImageProcessor

class OCRVisionTower(nn.Module):
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
    
    def resize_image(self, image):
        # transform image to (224,224)
        return transforms.Resize((224, 224))(image)

    def load_model(self):
        self.image_processor = LayoutLMv3ImageProcessor.from_pretrained("microsoft/layoutlmv3-large")

        # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.vision_tower = LayoutLMv3Model.from_pretrained("microsoft/layoutlmv3-large")
        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
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
        # print('OCR: rank:', os.environ['LOCAL_RANK'], 'images:', images.device, images.dtype, 'self:', self.device, self.dtype)
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_feature = self.feature_select(image_forward_out).to(dtype=image.dtype)
                image_features.append(image_feature)
        else:
            # images = self.resize_image(images)
            image_forward_outs = self.vision_tower(pixel_values=images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(dtype=images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return 1024

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        return 196