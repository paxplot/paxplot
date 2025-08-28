"""Paxplot modules"""
from .core import pax_parallel, PaxFigure
from .controller import PaxController
from .datasets import *
from . import data_managers

# Main interface - users should use PaxController
__all__ = [
    'PaxController',  # Main interface
    'pax_parallel',   # Legacy interface
    'PaxFigure',      # Legacy interface
]
