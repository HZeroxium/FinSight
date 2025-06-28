from .data_loader import FinancialDataLoader, FinancialDataset
from .feature_engineering import FeatureEngineering


# Convenience functions
def create_financial_data_loader(config):
    """Create a financial data loader instance"""
    return FinancialDataLoader(config)


def create_feature_engineering(config):
    """Create a feature engineering instance"""
    return FeatureEngineering(config)


__all__ = [
    "FinancialDataLoader",
    "FinancialDataset",
    "FeatureEngineering",
    "create_financial_data_loader",
    "create_feature_engineering",
]
