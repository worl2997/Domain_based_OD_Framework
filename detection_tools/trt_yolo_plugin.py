import ctypes
import sys
import os
import time
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
from utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
import torch
from detection_tools.utils import post_processing
import pycuda.autoinit

# Simple helper data class that's a little nicer to use than a 2-tuple.
def GiB(val):
    return val * 1 << 30

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

    def __del__(self):
        del self.device
        del self.host

def allocate_buffers(engine, batch_size):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:

        size = trt.volume(engine.get_binding_shape(binding)) * batch_size
        dims = engine.get_binding_shape(binding)

        # in case batch dimension is -1 (dynamic)
        if dims[0] < 0:
            size *= -1
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=1, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class Trt_yolo(object):

    def __init__(self, engine_path, num_classes, img_size):
        self.engine = engine_path
        self.num_classes = num_classes
        #self.letter_box = letter_box
        self.IN_IMAGE_H,self.IN_IMAGE_W = img_size
        # ????????? multi-batch??? ??????????????? ????????? ????????????
        self.inference_fn = do_inference
        self.trt_logger = trt.Logger()
        trt.init_libnvinfer_plugins(self.trt_logger, '')

        self.engine = self.get_engine()
        #self.input_shape = get_input_shape(self.engine)
        try:
            self.context = self.engine.create_execution_context()
            self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine, 1)
            self.context.set_binding_shape(0, (1, 3, self.IN_IMAGE_W, self.IN_IMAGE_W))

        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e

    def __del__(self):
        # Cuda memory free
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, image_src, conf_thresh=0.4, nms_thresh=0.6):
        resized = cv2.resize(image_src, (self.IN_IMAGE_W, self.IN_IMAGE_H), interpolation=cv2.INTER_LINEAR)
        img_in = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        img_in = np.ascontiguousarray(img_in)
        self.inputs[0].host = img_in # ???????????? ????????? ?????? resize??? ??????

        trt_outputs = do_inference(self.context, bindings=self.bindings, inputs=self.inputs, outputs=self.outputs, stream=self.stream)
        outputs = torch.from_numpy(trt_outputs[0])
        outputs = outputs.view(1,-1,  self.num_classes + 5)
        boxes = non_max_suppression(outputs, conf_thresh, nms_thresh)
        # nms??? ?????? boxes, scores, classes??? ?????? -> shape : (detected_instance_number , boxes, socres, classes)
        return boxes

    def get_engine(self):
        print("Reading engine from file {}".format(self.engine))
        with open(self.engine, 'rb') as f, trt.Runtime(self.trt_logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
