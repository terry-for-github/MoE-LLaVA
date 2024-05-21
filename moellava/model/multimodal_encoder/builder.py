import os
from .clip_encoder import CLIPVisionTower
from .dino_encoder import DINOVisionTower
from .ocr_encoder import OCRVisionTower
# from .graph_encoder import SGVisionTower
import transformers
a, b, c = transformers.__version__.split('.')[:3]
if a == '4' and int(b) >= 37:
    from .siglip_encoder import SiglipVisionTower
# from .languagebind import LanguageBindImageTower, LanguageBindVideoTower

# ============================================================================================================

def build_image_tower(image_tower_cfg, load_model='clip', **kwargs):
    image_tower = getattr(image_tower_cfg, 'mm_image_tower', getattr(image_tower_cfg, 'image_tower', None))
    is_absolute_path_exists = os.path.exists(image_tower)
    if (image_tower if not is_absolute_path_exists else os.path.basename(image_tower)).startswith("openai") or \
        (image_tower if not is_absolute_path_exists else os.path.basename(image_tower)).startswith("laion"):
        if load_model == "clip":
            return CLIPVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
        elif load_model == "dino":
            return DINOVisionTower(image_tower, args=image_tower_cfg, **kwargs)
        elif load_model == 'ocr':
            return OCRVisionTower(image_tower, args=image_tower_cfg, **kwargs)
        # elif load_model == 'graph':
        #     return SGVisionTower(image_tower, args=image_tower_cfg, **kwargs)
    if (image_tower if not is_absolute_path_exists else os.path.basename(image_tower)).startswith("google"):
        if load_model == "clip":
            return SiglipVisionTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)
        elif load_model == "dino":
            return DINOVisionTower(image_tower, args=image_tower_cfg, **kwargs)
        elif load_model == 'ocr':
            return OCRVisionTower(image_tower, args=image_tower_cfg, **kwargs)
        # elif load_model == 'graph':
        #     return SGVisionTower(image_tower, args=image_tower_cfg, **kwargs)

    if (image_tower if not is_absolute_path_exists else os.path.basename(image_tower)).endswith('LanguageBind_Image'):
        return LanguageBindImageTower(image_tower, args=image_tower_cfg, cache_dir='./cache_dir', **kwargs)

    raise ValueError(f'Unknown image tower: {image_tower}')

def build_video_tower(video_tower_cfg, **kwargs):
    video_tower = getattr(video_tower_cfg, 'mm_video_tower', getattr(video_tower_cfg, 'video_tower', None))
    if video_tower.endswith('LanguageBind_Video_merge'):
        return LanguageBindVideoTower(video_tower, args=video_tower_cfg, cache_dir='./cache_dir', **kwargs)
    raise ValueError(f'Unknown video tower: {video_tower}')
# ============================================================================================================
