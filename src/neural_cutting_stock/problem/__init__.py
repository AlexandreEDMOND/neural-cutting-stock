"""Problem representation and validation for 1D Cutting Stock instances."""

from .instance import CuttingStockInstance
from .multi_format import MULTI_STOCK_FORMAT_SCHEMA_VERSION, MultiFormatCuttingStockInstance

AnyCuttingStockInstance = CuttingStockInstance | MultiFormatCuttingStockInstance
"""Either cutting-stock form accepted by the classical solver components."""

__all__ = [
    "AnyCuttingStockInstance",
    "CuttingStockInstance",
    "MULTI_STOCK_FORMAT_SCHEMA_VERSION",
    "MultiFormatCuttingStockInstance",
]
