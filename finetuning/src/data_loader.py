# data_loader.py
from typing import Optional, List
from pydantic import BaseModel, Field
from datasets import load_dataset, DatasetDict, Dataset


class DataLoaderConfig(BaseModel):
    """
    Configuration for loading OHLCV data from CSV.
    """

    csv_path: str = Field(..., description="Path to the CSV file containing OHLCV data")
    symbol: Optional[str] = Field(
        None, description="Filter by asset symbol, e.g., 'BTCUSDT'"
    )
    timeframe: Optional[str] = Field(
        None, description="Filter by timeframe, e.g., '1d', '1h'"
    )
    exchange: Optional[str] = Field(None, description="Filter by exchange if specified")


class DataLoader:
    """
    Loads and preprocesses raw OHLCV CSV into a HuggingFace Dataset.
    """

    def __init__(self, cfg: DataLoaderConfig):
        self.cfg = cfg

    def load(self) -> DatasetDict:
        ds = load_dataset("csv", data_files={"raw": self.cfg.csv_path})
        ds = ds["raw"]
        # filter by symbol/timeframe/exchange if provided
        if self.cfg.symbol:
            ds = ds.filter(lambda ex: ex["symbol"] == self.cfg.symbol)
        if self.cfg.timeframe:
            ds = ds.filter(lambda ex: ex["timeframe"] == self.cfg.timeframe)
        if self.cfg.exchange:
            ds = ds.filter(lambda ex: ex["exchange"] == self.cfg.exchange)
        ds = ds.sort("timestamp")
        # split train/val/test: 80/10/10
        n = len(ds)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        return DatasetDict(
            {
                "train": ds.select(range(0, train_end)),
                "validation": ds.select(range(train_end, val_end)),
                "test": ds.select(range(val_end, n)),
            }
        )
