"""Problem representation and validation for 1D Cutting Stock instances."""

from .instance import CuttingStockInstance
from .multi_format import MULTI_STOCK_FORMAT_SCHEMA_VERSION, MultiFormatCuttingStockInstance

__all__ = [
    "CuttingStockInstance",
    "MULTI_STOCK_FORMAT_SCHEMA_VERSION",
    "MultiFormatCuttingStockInstance",
]
