"""Cấu hình pytest dùng chung cho worker.

Cổng gác nhóm `gpu_acceptance`: các bài kiểm THẬT chỉ GPU mới chứng minh được (cư trú
VRAM, dịch Qwen, /process e2e). Chúng KHÔNG BAO GIỜ được chạy — và tuyệt đối KHÔNG được
"pass giả" — trên hộp không đủ năng lực. Một bài gpu_acceptance mà xanh trên CPU-mock sẽ
nói dối rằng phần cứng đã kiểm chứng (vi phạm No-Fake-Success). Vì vậy skip trừ khi CẢ BA đúng:
  1) RUN_GPU_ACCEPT=1  — người chạy CHỦ Ý bật (không vô tình dính khi chạy toàn bộ suite);
  2) torch.cuda.is_available() — có GPU THẬT;
  3) compute capability >= 7.0 (Volta+) — GPU chạy được ĐÚNG đường production (float16).
     Đường suy luận production ép `compute_type="float16"` (asr_service) + trọng số fp16 —
     GPU Pascal (vd GTX 1070, cc 6.1) KHÔNG hỗ trợ fp16 hiệu quả -> model load HARD-ERROR,
     không bao giờ chứng minh được gì. Skip sạch (có lý do rõ) trung thực hơn error khó hiểu,
     và vẫn KHÔNG giả xanh (skip != pass). Hộp mục tiêu 24GB (Volta/Ampere/Ada, cc>=7) sẽ chạy.

pyproject `addopts = -m 'not gpu_acceptance'` đã loại nhóm này khỏi lần chạy MẶC ĐỊNH.
Lớp gác dưới đây là dây an toàn thứ hai: bắt cả trường hợp ai đó chủ động gọi
`pytest -m gpu_acceptance` trên máy KHÔNG có GPU đủ mạnh (CPU-only, hoặc GPU dev quá yếu như
GTX 1070 8GB) -> skip, không giả xanh và không error khó hiểu.
"""
import os

import pytest


def _gpu_acceptance_enabled() -> bool:
    if os.environ.get("RUN_GPU_ACCEPT") != "1":
        return False
    try:
        import torch  # CPU torch vẫn import được; chỉ .cuda mới phân biệt phần cứng thật
    except Exception:
        return False
    try:
        if not torch.cuda.is_available():
            return False
        # Ép đúng đường production: cần fp16 hiệu quả (Volta+ / cc>=7). Pascal (cc 6.x) rớt.
        major = torch.cuda.get_device_capability(0)[0]
        return major >= 7
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    if _gpu_acceptance_enabled():
        return  # đủ điều kiện -> cho chạy thật trên hộp GPU
    skip = pytest.mark.skip(
        reason="gpu_acceptance: cần RUN_GPU_ACCEPT=1 và torch.cuda.is_available() (chỉ hộp GPU)"
    )
    for item in items:
        if "gpu_acceptance" in item.keywords:
            item.add_marker(skip)
