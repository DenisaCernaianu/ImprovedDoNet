import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--annTrainFile', type=str, required=True)
parser.add_argument('--imgTrainFile', type=str, required=True)
parser.add_argument('--annValFile',   type=str, required=True)
parser.add_argument('--imgValFile',   type=str, required=True)
parser.add_argument('--configFile',   type=str,
    default="COCO-InstanceSegmentation/mask_triple_branch_rcnn_R_50_FPN_3x_heads_attention.yaml")
parser.add_argument('--outputDir',    type=str, required=True)
args = parser.parse_args()



register_coco_instances("cnseg_train", {}, args.annTrainFile, args.imgTrainFile)
register_coco_instances("cnseg_val",   {}, args.annValFile,   args.imgValFile)


cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(args.configFile)

cfg.DATASETS.TRAIN = ("cnseg_train",)
cfg.DATASETS.TEST  = ("cnseg_val",)   

cfg.MODEL.MASK_ON = True
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1   
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)

cfg.SOLVER.BASE_LR       = 0.00025
cfg.SOLVER.STEPS         = (40000, 50000)
cfg.SOLVER.MAX_ITER      = 60000
cfg.SOLVER.WEIGHT_DECAY  = 0.0001
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.CHECKPOINT_PERIOD = 10000

cfg.TEST.EVAL_PERIOD = 10000
cfg.VIS_PERIOD       = 500

cfg.INPUT.MIN_SIZE_TRAIN = (512, 576, 640, 704, 768, 832, 896)

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5


cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_TYPE           = "ROIAlignV2"
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONV_DIM              = 256
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.NORM                  = ""
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CLS_AGNOSTIC_MASK     = True
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.CONV_DIM        = 256
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.NORM            = ""
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.CLS_AGNOSTIC_MASK = True
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.REFINEMENT            = True
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.PURE_BRANCH           = False



cfg.OUTPUT_DIR = args.outputDir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg)


evaluator = COCOEvaluator(
    "cnseg_val", cfg, False,
    output_dir=cfg.OUTPUT_DIR + "/Output/cnseg_val"
)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return evaluator

cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)

if torch.cuda.is_available():
    torch.cuda.empty_cache()

trainer.train()
