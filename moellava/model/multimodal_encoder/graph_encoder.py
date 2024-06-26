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


from maskrcnn_benchmark.modeling.roi_heads.relation_head.utils_motifs import obj_edge_vectors, rel_vectors
from collections import OrderedDict
VG_ACTIONS = ['background', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to',
              'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on',
              'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on',
              'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on',
              'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using',
              'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

VG_Classes = ['background', 'airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed',
              'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch',
              'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter',
              'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face',
              'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove',
              'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean',
              'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men',
              'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people',
              'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot',
              'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short',
              'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street',
              'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck',
              'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing',
              'wire', 'woman', 'zebra']



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        return super().forward(x)
        # orig_type = x.dtype
        # ret = super().forward(x.type(torch.float32))
        # return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        self.mlp = nn.Linear(width, 4096)

    def forward(self, x: torch.Tensor):
        return self.mlp(self.resblocks(x))


class GraphProcessor(object):
    def __init__(self, cfg):
        # self.crop_size = {"height": cfg.INPUT.MIN_SIZE_TRAIN, "width": cfg.INPUT.MIN_SIZE_TRAIN}
        self.crop_size = {"height": 600, "width": 600}
        self.image_mean = [102.9801 / 255, 115.9465 / 255, 122.7717 / 255]
        self.image_std = [1., 1., 1.]
        self.transforms = build_transforms(cfg)

    def preprocess(self, img):
        box = torch.tensor([[34, 323, 30, 40]])

        w, h = img.size[0], img.size[1]

        # img.save('demo.png')
        proposal = BoxList(box, (w, h), 'xyxy')  # xyxy

        img, proposal = self.transforms(img, proposal)
        return img

class SGVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()
        self.is_loaded = False
        self.is_fp16 = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.sg_home = '/home/pcl/'
        self.sgg_config_path = self.sg_home + 'sgg_graph_encoder/configs/e2e_merge_relation_X_101_32_8_FPN_1x.yaml'
        self.sgg_model_dir = self.sg_home + 'trans_baseline' ###'/home/pcl/upload_causal_motif_sgdet'
        self.sgg_model_path = self.sg_home + 'trans_baseline/model_final.pth'
        cache_file = torch.load(self.sg_home + 'trans_baseline/VG_stanford_filtered_with_attribute_train_statistics.cache')
        self.rel_classes = cache_file['rel_classes']
        self.obj_classes = cache_file['obj_classes']
        self.GLOVE_DIR = self.sg_home + 'glove'

        cfg.merge_from_file(self.sgg_config_path)
        cfg.OUTPUT_DIR = self.sgg_model_dir
        cfg.GLOVE_DIR = self.GLOVE_DIR
        cfg.MODEL.ROI_HEADS.DETECTIONS_PER_IMG = 72 ## 16
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX = False
        cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL = False
        cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR = 'TransformerPredictor'
        cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS = True
        cfg.MODEL.ROI_RELATION_HEAD.RETURN_GRAPH_EMBEDDING = False
        cfg.freeze()
        self.cfg = cfg
        self.img_mean, self.img_std = np.array([0.48145466, 0.4578275, 0.40821073]), np.array(
            [0.26862954, 0.26130258, 0.27577711])

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

        self.embed_dim = cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        rel_embed_vecs = rel_vectors(self.rel_classes, wv_dir=cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(len(self.obj_classes), self.embed_dim)
        self.rel_embed = nn.Embedding(len(self.rel_classes), self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(obj_embed_vecs, non_blocking=True)
            self.rel_embed.weight.copy_(rel_embed_vecs, non_blocking=True)

        width = self.embed_dim
        num_layer = 4
        num_head = 8
        self.transformer = Transformer(width, num_layer, num_head)
        self.transformer.to(dtype=torch.float16)

    def load_model(self):

        self.sgg_model = build_detection_model(self.cfg)
        checkpointer = DetectronCheckpointer(self.cfg, self.sgg_model, save_dir=self.cfg.OUTPUT_DIR)
        _ = checkpointer.load(self.sgg_model_path)
        self.transforms = build_transforms(self.cfg, False)
        # self.clip_vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        self.image_processor = GraphProcessor(cfg)
        self.sgg_model.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def encode_graphs_(self, img):
        # import os
        # if os.environ['LOCAL_RANK'] == '0':
        #     print('graph:', img.size(), img.dtype, img.device)
        # if self.is_fp16 == False:
        #     self.sgg_model.to(dtype=torch.float16)
        #     self.is_fp16 = True
        # img = Image.open(image_path).convert("RGB")
        # box = torch.tensor([[34, 323, 30, 40]]).to(self.device)
        # w, h = img.size[0], img.size[1]
        # proposal = BoxList(box, (w, h), 'xyxy')  # xyxy
        #
        # img, proposal = self.transforms(img, proposal)
        #
        # proposal = proposal.to(self.device)
        #
        # image = to_image_list([img], 32).to(self.device)
        # para = next(self.sgg_model.parameters())
        # img = self.image_processor(img)
        self.sgg_model.eval()
        with torch.no_grad():
            graph_embeddings = self.sgg_model(img, [[]], [], [0], [[]])
            print(graph_embeddings)

        return graph_embeddings

    # @torch.no_grad()
    def encode_graphs(self, img):
        # import os
        # if os.environ['LOCAL_RANK'] == '0':
        #     print('graph:', img.size(), img.dtype, img.device)
        # if self.is_fp16 == False:
        #     self.sgg_model.to(dtype=torch.float16)
        #     self.is_fp16 = True
        # img = Image.open(image_path).convert("RGB")
        # box = torch.tensor([[34, 323, 30, 40]]).to(self.device)
        # w, h = img.size[0], img.size[1]
        # proposal = BoxList(box, (w, h), 'xyxy')  # xyxy
        #
        # img, proposal = self.transforms(img, proposal)
        #
        # proposal = proposal.to(self.device)
        #
        # image = to_image_list([img], 32).to(self.device)
        # para = next(self.sgg_model.parameters())
        # img = self.image_processor(img)
        self.sgg_model.eval()
        with torch.no_grad():
            results = self.sgg_model(img, [[]], [], [0], [[]])

        graph_embeddings = []
        for result in results:
            pred_labels = result.get_field('pred_labels')
            rel_pair_idxs = result.get_field('rel_pair_idxs')
            rel_labels = result.get_field('pred_rel_labels')
            pair_pred = torch.stack((pred_labels[rel_pair_idxs[:, 0]], pred_labels[rel_pair_idxs[:, 1]]), dim=1)
            head_emb = self.obj_embed(pair_pred[:, 0])
            rel_emb = self.rel_embed(rel_labels)
            tail_emb = self.obj_embed(pair_pred[:, 1])
            triple_emb = head_emb + rel_emb + tail_emb
            knowledge_emb = self.transformer(triple_emb)
            knowledge_emb = torch.mean(knowledge_emb, dim=0).unsqueeze(0)
            graph_embeddings.append(knowledge_emb)
        graph_embeddings = torch.cat(graph_embeddings, dim=0)

        return graph_embeddings

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
        # image_features = []
        # for image in images:
        #     image_features.append(self.encode_graphs(image))
        # image_features = torch.stack(image_features).to(dtype=torch.bfloat16)
        # return image_features
        if type(images) is list:
            # image_features = []
            images = to_image_list(images, 32).to(self.device)
            image_features = self.encode_graphs(images.to(device=self.device, dtype=self.dtype))
            image_features = image_features.unsqueeze(1)
            # for image in images:
            #     images = to_image_list(img, 32).to(self.device)
            #     # image_features.append(image_feature)
            #     image_feature = self.encode_graphs(image.to(device=self.device, dtype=self.dtype))
            #     print(image_feature.shape)
            #     # image_feature = self.feature_select(image_forward_out).to(dtype=image.dtype)
            #     image_features.append(image_feature)

        else:
            image_forward_outs = self.encode_graphs(images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(image_forward_outs).to(dtype=images.dtype)

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