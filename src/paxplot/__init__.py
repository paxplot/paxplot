"""Paxplot modules"""
from .legacy.core import pax_parallel, PaxFigure
from .legacy.controller import PaxController
from .datasets import *
from .legacy import data_managers

# Main interface - users should use PaxController
__all__ = [
    'PaxController',  # Main interface
    'pax_parallel',   # Legacy interface
    'PaxFigure',      # Legacy interface
]
