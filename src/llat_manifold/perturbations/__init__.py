"""Perturbation (injection) system: hooks + registry."""
from .base import Perturbation, StepContext
from .registry import build_perturbation, register

__all__ = ["Perturbation", "StepContext", "build_perturbation", "register"]
