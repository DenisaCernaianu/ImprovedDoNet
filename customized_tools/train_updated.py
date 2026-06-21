import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from amodal_evaluation import DuringTrainAmodalEvaluator
import argparse
import torch
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--annTrainFile', type=str, required=True, help="path to the train json file")
parser.add_argument('--imgTrainFile', type=str, required=True, help="path to the train image file")
parser.add_argument('--annValFile', type=str, required=True, help="path to the validation json file")
parser.add_argument('--imgValFile', type=str, required=True, help="path to the validation image file")
parser.add_argument('--configFile', type=str, default="COCO-InstanceSegmentation/mask_triple_branch_rcnn_R_50_FPN_3x_heads_attention.yaml", help="path to the config file")
parser.add_argument('--outputDir', type=str, required=True, help="path to the output directory, note that this is the directory that stores the weights file, and the evaluation results will be stored in <outputDir>/results")

args = parser.parse_args()

annTrainFile = args.annTrainFile
imgTrainFile = args.imgTrainFile    
annValFile = args.annValFile
imgValFile = args.imgValFile
configFile = args.configFile
outputDir = args.outputDir

register_coco_instances("amodal_coco_train", {},annTrainFile , imgTrainFile)
register_coco_instances("amodal_coco_val", {}, annValFile, imgValFile)

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(configFile)
cfg.DATASETS.TRAIN = ("amodal_coco_train",)
cfg.DATASETS.TEST = ("amodal_coco_val",)
cfg.MODEL.MASK_ON=True
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 


cfg.SOLVER.BASE_LR = 0.00025 
cfg.SOLVER.STEPS = (40000, 50000) 
cfg.SOLVER.MAX_ITER = 60000
cfg.SOLVER.WEIGHT_DECAY = 0.0001 
cfg.SOLVER.IMS_PER_BATCH = 2 #INITAL 4 
cfg.SOLVER.CHECKPOINT_PERIOD = 10000
cfg.TEST.EVAL_PERIOD = 10000
cfg.VIS_PERIOD = 500
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONV_DIM=256
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.NORM=""
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CLS_AGNOSTIC_MASK=True
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.CONV_DIM=256
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.NORM=""
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.CLS_AGNOSTIC_MASK=True
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.REFINEMENT=True
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.PURE_BRANCH = False
cfg.OUTPUT_DIR = outputDir
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
print(cfg)

evaluator = COCOEvaluator("amodal_coco_val", cfg, False, output_dir=cfg.OUTPUT_DIR + "/Output/" + cfg.DATASETS.TEST[0])

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return evaluator

trainer = Trainer(cfg)

trainer.resume_or_load(resume=False)
if torch.cuda.is_available():
   torch.cuda.empty_cache()
trainer.train()

