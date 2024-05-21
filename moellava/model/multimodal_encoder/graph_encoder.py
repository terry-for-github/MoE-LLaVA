import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
# from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


class SGVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.is_fp16 = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.sgg_config_path = '/home/pcl/sgg_graph_encoder/configs/e2e_relation_X_101_32_8_FPN_1x.yaml'
        self.sgg_model_dir = '/home/pcl/sgdet_trans_baseline' ###'/home/pcl/upload_causal_motif_sgdet'
        self.GLOVE_DIR = '/home/pcl/glove'

        cfg.merge_from_file(self.sgg_config_path)
        cfg.OUTPUT_DIR = self.sgg_model_dir
        cfg.GLOVE_DIR = self.GLOVE_DIR
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
        cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = 'TransformerPredictor'
        cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
        cfg.MODEL.ROI_RELATION_HEAD.RETURN_GRAPH_EMBEDDING = True
        cfg.freeze()
        self.cfg = cfg
        self.img_mean, self.img_std = np.array([0.48145466, 0.4578275, 0.40821073]), np.array(
            [0.26862954, 0.26130258, 0.27577711])

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):

        self.sgg_model = build_detection_model(self.cfg)
        checkpointer = DetectronCheckpointer(self.cfg, self.sgg_model, save_dir=self.cfg.OUTPUT_DIR)
        _ = checkpointer.load(self.cfg.MODEL.WEIGHT)
        self.transforms = build_transforms(self.cfg, False)
        # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)

        self.sgg_model.requires_grad_(False)
        for name, param in self.sgg_model.named_parameters():
            if 'rnn' in name:
                param.data = param.data.float()
        self.sgg_model.eval()

        self.is_loaded = True

    @torch.no_grad()
    def encode_graphs(self, img):
        # if self.is_fp16 == False:
        #     self.sgg_model.to(dtype=torch.float16)
        #     self.is_fp16 = True
        # img = Image.open(image_path).convert("RGB")
        box = torch.tensor([[34, 323, 30, 40]]).to(self.device)
        w, h = img.size[0], img.size[1]
        proposal = BoxList(box, (w, h), 'xyxy')  # xyxy

        img, proposal = self.transforms(img, proposal)

        proposal = proposal.to(self.device)

        image = to_image_list([img], 32).to(self.device)
        # para = next(self.sgg_model.parameters())
        self.sgg_model.eval()
        with torch.no_grad():
            graph_embeddings = self.sgg_model(image, [[]], [proposal], [0], [[]])

        return graph_embeddings

    def _inv_img_tensors(self, img):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = (img * self.img_std * 255 + self.img_mean * 255).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')
        return img

    @torch.no_grad()
    def forward(self, images):
        # if type(images) is list:
        #     image_features = []
        #     for image in images:
        #         # image = self._inv_img_tensors(image)
        #         image_feature = self.encode_graphs(image)
        #         image_features.append(image_feature)
        # else:
            # images = self._inv_img_tensors(images)
        image_features = []
        for image in images:
            image_features.append(self.encode_graphs(image))
        image_features = torch.stack(image_features).to(dtype=torch.bfloat16)
        return image_features

    @property
    def hidden_size(self):
        #return self.config.hidden_size
        return 4096

    @property
    def dtype(self):
        return next(self.sgg_model.parameters()).dtype

    @property
    def device(self):
        return next(self.sgg_model.parameters()).device

    @property
    def config(self):
        if self.is_loaded:
            return self.sgg_model.config
        else:
            return self.cfg_only

    @property
    def num_patches(self):
        #return (self.config.image_size // self.config.patch_size) ** 2
        raise ValueError('Scene Graph Encoder has no num_patches')

#
# sgg_config_path = '/home/pcl/sgg_graph_encoder/configs/e2e_relation_X_101_32_8_FPN_1x.yaml'
# sgg_model_dir = '/home/pcl/upload_causal_motif_sgdet'
# image_path = '/home/pcl/713702_ori.jpg'
# img = Image.open(image_path).convert("RGB")
# print(img.size[0], img.size[1])
# sgg_graph_encoder = SGVisionTower(sgg_config_path, sgg_model_dir)
# graph_embeddings = sgg_graph_encoder(img)
# print(graph_embeddings.shape)