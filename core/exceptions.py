# -*- coding: utf-8 -*-
"""统一异常类型。"""
from __future__ import annotations


class DependencyMissing(ImportError):
    """可选重依赖缺失（ultralytics / onnxruntime / av / scrcpy）。

    设计原则：轻依赖可插拔 —— 缺库时调用方应能优雅降级
    （例如换用 MockFastDetector），而不是让整个进程崩溃。
    """

    def __init__(self, module: str, purpose: str, install: str):
        self.module = module
        self.purpose = purpose
        self.install = install
        super().__init__(
            f"[{module}] 未安装 —— {purpose}。"
            f"安装: {install}；或使用 Mock 降级实现。"
        )
