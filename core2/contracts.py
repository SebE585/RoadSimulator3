# core2/contracts.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Result:
    ok: bool = True
    message: str = ""