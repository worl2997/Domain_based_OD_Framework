from __future__ import division
from parser import *
from OID_tools.bounding_boxes import *
from train import *
import os
import sys

ROOT_DIR = os.path.abspath(os.path.dirname(sys.argv[0]))
DEFAULT_DATA_DIR = os.path.join(ROOT_DIR, 'data','custom')
DATA_FILE_DIR = os.path.join(ROOT_DIR,'config','custom_data')
cfg_path =  os.path.join(ROOT_DIR, 'config','custom_cfg')
model_save_path = os.path.join(ROOT_DIR,'weights', 'custom_weight')

if __name__ == '__main__':
    args = parse_arguments()
    if args.command == 'downloader':
        domain_groups = bounding_boxes_images(args, ROOT_DIR, DEFAULT_DATA_DIR)
        print(domain_groups)

    elif args.command == 'train':
        data_files = os.path.join(DATA_FILE_DIR, args.domain + '.data')
        data_config = parse_data_config(data_files)
        train_path = data_config['train']
        valid_path = data_config["valid"]
        class_names = load_classes(data_config["names"])
        if args.cfg:
            model_cfg = args.cfg
        else:
            model_cfg = get_group_cfg(cfg_path, args.domain, args.model, data_config['classes'])
        train(args, True, train_path, valid_path, class_names, model_cfg, model_save_path)

