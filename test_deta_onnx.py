from pathlib import Path
from optimum.exporters import TasksManager
from optimum.exporters.onnx import export
from transformers import AutoModelForObjectDetection

base_model = AutoModelForObjectDetection.from_pretrained("nielsr/deta-resnet-50")

onnx_path = Path("model.onnx")
onnx_config_constructor = TasksManager.get_exporter_config_constructor("onnx", base_model)
onnx_config = onnx_config_constructor(base_model.config)

onnx_inputs, onnx_outputs = export(base_model, onnx_config, onnx_path, onnx_config.DEFAULT_ONNX_OPSET)
