import sys
import onnx
import os
import argparse
import numpy as np
import cv2
import onnxruntime
import torch
from models import Darknet
from models import load_model



def transform_to_onnx(pre_weights,model_cfg, batch_size, IN_IMAGE_H, IN_IMAGE_W):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet(model_cfg).to(device)
    file_name = model_cfg.split('/')[-1].split('.')[0]

    # If pretrained weights are specified, start from checkpoint or weight file
    if pre_weights:
        if pre_weights.endswith(".pth"):
            # Load checkpoint weights
            print('pre_wegiht:',pre_weights)
            model.load_state_dict(torch.load(pre_weights, map_location=device))
            print('load model ')
        else:
            # Load darknet weights
            model.load_darknet_weights(pre_weights)

    input_names = ["input"]
    output_names = ["outputs"]
    dynamic = False  # for support dynamic batch, not supported yet

    # model.eval()


    if batch_size <= 0:
        dynamic = True

    if dynamic:
        x = torch.randn((1, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).to(device)
        onnx_file_name = file_name + "_dynamic.onnx"
        dynamic_axes = {"input": {0: "batch_size"}, "boxes": {0: "batch_size"}, "confs": {0: "batch_size"}}
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes)

        print('Onnx model exporting done')
        return onnx_file_name

    else:
        x = torch.randn((batch_size, 3, IN_IMAGE_H, IN_IMAGE_W), requires_grad=True).to(device)
        onnx_file_name = file_name + "_static.onnx"
        # Export the model
        print('Export the onnx model ...')
        torch.onnx.export(model,
                          x,
                          onnx_file_name,
                          export_params=True,
                          opset_version=11,
                          do_constant_folding=True,
                          input_names=input_names, output_names=output_names,
                          dynamic_axes=None)

        print('Onnx model exporting done')
        return onnx_file_name


def main(weight_file, cfg_file, batch_size, IN_IMAGE_H, IN_IMAGE_W):
    if batch_size <= 0:
        # not developed yet
        onnx_path_demo = transform_to_onnx(weight_file, cfg_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        # Transform to onnx for demo
        onnx_path_demo = transform_to_onnx(weight_file,cfg_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)

if __name__ == '__main__':
    print("Converting to onnx and running demo ...")
    if len(sys.argv) == 6:

        weight_file = sys.argv[1]
        cfg_file = sys.argv[2]
        batch_size = int(sys.argv[3])
        IN_IMAGE_H = int(sys.argv[4])
        IN_IMAGE_W = int(sys.argv[5])

        main(weight_file,cfg_file, batch_size, IN_IMAGE_H, IN_IMAGE_W)
    else:
        print('Please run this way:\n')
        print('  python demo_onnx.py <weight_file> <cfg_file> <batch_size> <IN_IMAGE_H> <IN_IMAGE_W>')