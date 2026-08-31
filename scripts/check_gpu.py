import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch_directml
from device import detect_device, report_device

print("torch", torch.__version__)
print("dml available", torch_directml.is_available())
print("dml count", torch_directml.device_count())
for i in range(torch_directml.device_count()):
    print(f"  [{i}]", torch_directml.device_name(i))

info = detect_device()
report_device(info)
x = torch.ones(4, device=info.device)
print("tensor device", x.device, "sum", float(x.sum()))
