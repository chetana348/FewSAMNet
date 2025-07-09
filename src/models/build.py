from src.models.network1 import *
from src.models.segment_anything.build_sam import sam_model_registry
from src.config import Config
import torch


def build_resSAM(need_ori_checkpoint, model_checkpoint):
    return _build_sam(
        type=50,
        sam_model=sam_model_registry['vit_b_256'],
        ori_checkpoint=need_ori_checkpoint,
        model_checkpoint=model_checkpoint,
    )


model_registry = {
    'resSAM': build_resSAM,
}


def _build_sam(type, 
               sam_model, 
               ori_checkpoint: bool, 
               model_checkpoint: str):
    if model_checkpoint is not None and ori_checkpoint == False:
        model_checkpoint = torch.load(model_checkpoint)
        if 'model' in model_checkpoint.keys():
            model_checkpoint = model_checkpoint['model']
        model = MultiStreamSegmentor(sam_model=sam_model(None), use_pretrained=False)
        model.load_state_dict(model_checkpoint)
    elif ori_checkpoint == True: 
        model = MultiStreamSegmentor(sam_model=sam_model(config_dict['checkpoint_path']), use_pretrained=True)
    else:
        model = MultiStreamSegmentor(sam_model=sam_model(None), use_pretrained=False)
    return model
