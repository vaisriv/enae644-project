"""Adversarial motion planning system."""

__version__ = "0.1.0"

__all__ = ["main", "train", "deceptive", "interceptor", "shared", "simulation", "data"]

from src.index import main
from src.training import train

from src import deceptive
from src import interceptor
from src import shared
from src import simulation
from src import data
