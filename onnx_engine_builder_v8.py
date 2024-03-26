import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Tuple, Union
import json

import onnx
import torch
import tensorrt as trt

metadata = {
    "description": "Model description",
    "author": "Sid Prabhakaran",
    "date": datetime.now().isoformat(),
    "version": "0.1",
    "license": "None",
    "docs": "https://docs.ultralytics.com",
    "stride": 32,
    "task": "detect",
    "batch": 1,
    "imgsz": [416, 416],
    "names": {"0": "Blue", "1": "Red"},
}  # model metadata


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx',
                        type=str,
                        required=True,
                        help='YOLOv8-TensorRT onnx file')
    parser.add_argument('--out',
                        type=str,
                        required=True,
                        help='Output engine file')
    args = parser.parse_args()
    return args


class EngineBuilder:
    def __init__(
        self,
        out_file: Union[str, Path],
        checkpoint: Union[str, Path],
        device: Optional[Union[str, int, torch.device]] = None) -> None:
        
        checkpoint = Path(checkpoint) if isinstance(checkpoint, str) else checkpoint
        assert checkpoint.exists(), f'{checkpoint} not found'
        assert checkpoint.suffix in ('.onnx'), f'invalid checkpoint format {checkpoint.suffix}'
        if isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device(f'cuda:{device}')

        self.checkpoint = checkpoint
        self.device = device
        self.out_file = out_file

    def build(
        self,
        fp16: bool = True,
        iou_thres: float = 0.2,
        conf_thres: float = 0.25,
        topk: int = 20,
        with_ultralytics_metadata: bool = True) -> None:

        logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(logger, namespace='')
        builder = trt.Builder(logger)
        config = builder.create_builder_config()
        config.max_workspace_size = torch.cuda.get_device_properties(self.device).total_memory
        flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flag)

        parser = trt.OnnxParser(network, logger)
        onnx_model = onnx.load(str(self.checkpoint))

        onnx_model.graph.node[-1].attribute[2].i = topk
        onnx_model.graph.node[-1].attribute[3].f = conf_thres
        onnx_model.graph.node[-1].attribute[4].f = iou_thres

        if not parser.parse(onnx_model.SerializeToString()):
            raise RuntimeError(
                f'failed to load ONNX file: {str(self.checkpoint)}')

        inputs = [network.get_input(i) for i in range(network.num_inputs)]
        outputs = [network.get_output(i) for i in range(network.num_outputs)]
        for inp in inputs:
            logger.log(trt.Logger.WARNING,
                f'input "{inp.name}" with shape: {inp.shape} '
                f'dtype: {inp.dtype}')
        for out in outputs:
            logger.log(trt.Logger.WARNING,
                f'output "{out.name}" with shape: {out.shape} '
                f'dtype: {out.dtype}')

        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        self.logger = logger
        self.builder = builder
        self.network = network

        with builder.build_engine(network, config) as engine, open(self.out_file, "wb") as f:
            if with_ultralytics_metadata:
                meta = json.dumps(metadata)
                f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
                f.write(meta.encode())
            f.write(engine.serialize())

def main(args):
    builder = EngineBuilder(args.out, args.onnx)
    builder.build()
    print("Done!")

if __name__ == '__main__':
    args = parse_args()
    main(args)