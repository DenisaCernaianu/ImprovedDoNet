from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.modeling.roi_heads import roi_heads
import os
import argparse
import torch
from torch import nn
import torch.nn.functional as F 
from detectron2.data import build_detection_test_loader, DatasetMapper, build_detection_train_loader, detection_utils as utils
from detectron2.data import datasets, DatasetCatalog, MetadataCatalog


register_coco_instances("isbi_test", {}, "datasets/isbi/annotations/isbi_test.json", "datasets/isbi/isbi_test")
register_coco_instances("isbi_train", {}, "datasets/isbi/annotations/isbi_train.json", "datasets/isbi/isbi_train")

parser = argparse.ArgumentParser()

parser.add_argument('--annTestFile', type=str, required=True, help="path to the test json file")
parser.add_argument('--imgTestFile', type=str, required=True, help="path to the test image file")
parser.add_argument('--configFile', type=str, default="COCO-InstanceSegmentation/mask_triple_branch_rcnn_R_50_FPN_3x_heads_attention.yaml", help="path to the config file")
parser.add_argument('--outputDir', type=str, required=True, help="path to the output directory, note that this is the directory that stores the weights file, and the evaluation results will be stored in <outputDir>/results")
parser.add_argument('--weightsFile', type=str, default="model_final.pth", help="name of the weights file, note that the absolute path is <outputDir>/<weightsFile>")

args = parser.parse_args()
    
'''
to run:
python evaluate.py 
    --annTestFile <path to the test json file> 
    --imgTestFile <path to the test image file> 
    --configFile <path to the config file> 
    --outputDir <path to the output directory>
    --weightsFile <name of the weights file>
'''
    
annTestFile = args.annTestFile
imgTestFile = args.imgTestFile
configFile = args.configFile
outputDir = args.outputDir
weightsFile = args.weightsFile
register_coco_instances("isbi_test", {}, annTestFile, imgTestFile)

cfg = get_cfg()
cfg.set_new_allowed(True)
cfg.merge_from_file(configFile)
cfg.MODEL.MASK_ON=True
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.9
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.POOLER_TYPE = "ROIAlignV2" 
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CONV_DIM=256 
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.NORM="" 
cfg.MODEL.ROI_TRIPLE_BRANCH_WHOLE_MASK_HEAD.CLS_AGNOSTIC_MASK=True 
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.CONV_DIM=256 
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.NORM="" 
cfg.MODEL.ROI_TRIPLE_BRANCH_OVERLAPPING_MASK_HEAD.CLS_AGNOSTIC_MASK=True 
cfg.OUTPUT_DIR = outputDir
cfg.DATASETS.TEST = ("isbi_test",)
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, weightsFile)


trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)

evaluator = COCOEvaluator("isbi_test", cfg, False,
                            output_dir=os.path.join(cfg.OUTPUT_DIR, "results"))

val_loader = build_detection_test_loader(cfg, "isbi_test")
inference_on_dataset(trainer.model, val_loader, evaluator)
