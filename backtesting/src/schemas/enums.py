# schemas/enums.py

from enum import Enum


class RepositoryType(Enum):
    """Available repository types"""

    CSV = "csv"
    MONGODB = "mongodb"
    INFLUXDB = "influxdb"
    TIMESCALEDB = "timescaledb"
