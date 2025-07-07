# strategies/strategy_factory.py

"""
Strategy Factory for creating strategy instances.
Implements Factory Pattern for strategy instantiation.
"""

from typing import Dict, Any, List
from ..schemas.backtesting_schemas import StrategyType, StrategyConfig
from ..interfaces.strategy import Strategy, StrategyError
from .moving_average_crossover import MovingAverageCrossoverStrategy
from .rsi_strategy import RSIStrategy
from .simple_buy_hold import SimpleBuyHoldStrategy
from .bollinger_bands import BollingerBandsStrategy
from .macd_strategy import MACDStrategy


class StrategyFactory:
    """
    Factory for creating trading strategy instances.

    Centralizes strategy creation and provides type safety.
    """

    _strategy_registry = {
        StrategyType.MOVING_AVERAGE_CROSSOVER: MovingAverageCrossoverStrategy,
        StrategyType.RSI_STRATEGY: RSIStrategy,
        StrategyType.SIMPLE_BUY_HOLD: SimpleBuyHoldStrategy,
        StrategyType.BOLLINGER_BANDS: BollingerBandsStrategy,
        StrategyType.MACD_STRATEGY: MACDStrategy,
    }

    @classmethod
    def create_strategy(cls, config: StrategyConfig) -> Strategy:
        """
        Create a strategy instance based on configuration.

        Args:
            config: Strategy configuration

        Returns:
            Strategy instance

        Raises:
            StrategyError: If strategy type is not supported
        """
        strategy_type = config.strategy_type

        if strategy_type not in cls._strategy_registry:
            available_strategies = list(cls._strategy_registry.keys())
            raise StrategyError(
                f"Unsupported strategy type: {strategy_type}. "
                f"Available strategies: {available_strategies}"
            )

        strategy_class = cls._strategy_registry[strategy_type]
        strategy = strategy_class(config)

        # Validate parameters
        strategy.validate_parameters()

        return strategy

    @classmethod
    def get_supported_strategies(cls) -> List[StrategyType]:
        """
        Get list of supported strategy types.

        Returns:
            List of supported strategy types
        """
        return list(cls._strategy_registry.keys())

    @classmethod
    def get_default_parameters(cls, strategy_type: StrategyType) -> Dict[str, Any]:
        """
        Get default parameters for a strategy type.

        Args:
            strategy_type: Strategy type

        Returns:
            Dictionary of default parameters

        Raises:
            StrategyError: If strategy type is not supported
        """
        defaults = {
            StrategyType.MOVING_AVERAGE_CROSSOVER: {
                "fast_period": 10,
                "slow_period": 30,
                "ma_type": "sma",
            },
            StrategyType.RSI_STRATEGY: {
                "rsi_period": 14,
                "oversold_threshold": 30,
                "overbought_threshold": 70,
            },
            StrategyType.BOLLINGER_BANDS: {
                "bb_period": 20,
                "bb_std_dev": 2.0,
                "bb_position": "mean_reversion",
            },
            StrategyType.MACD_STRATEGY: {
                "fast_period": 12,
                "slow_period": 26,
                "signal_period": 9,
                "histogram_threshold": 0.0,
            },
            StrategyType.SIMPLE_BUY_HOLD: {"entry_delay": 0},
        }

        if strategy_type not in defaults:
            raise StrategyError(f"No default parameters for strategy: {strategy_type}")

        return defaults[strategy_type]

    @classmethod
    def get_available_strategies(cls) -> List[str]:
        """
        Get list of available strategy names.

        Returns:
            List of strategy names as strings
        """
        return [strategy_type.value for strategy_type in cls._strategy_registry.keys()]

    @classmethod
    def is_strategy_available(cls, strategy_name: str) -> bool:
        """
        Check if a strategy is available.

        Args:
            strategy_name: Name of the strategy to check

        Returns:
            True if strategy is available, False otherwise
        """
        try:
            strategy_type = StrategyType(strategy_name)
            return strategy_type in cls._strategy_registry
        except ValueError:
            return False

    @classmethod
    def get_strategy_config(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration schema for a specific strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary containing strategy configuration schema

        Raises:
            StrategyError: If strategy is not available
        """
        if not cls.is_strategy_available(strategy_name):
            raise StrategyError(f"Strategy not available: {strategy_name}")

        strategy_type = StrategyType(strategy_name)
        strategy_class = cls._strategy_registry[strategy_type]

        # Create temporary instance to get configuration
        temp_config = StrategyConfig(strategy_type=strategy_type, parameters={})

        try:
            temp_strategy = strategy_class(temp_config)

            # Get parameter information
            required_params = temp_strategy.get_required_parameters()
            optional_params = temp_strategy.get_optional_parameters()
            param_descriptions = temp_strategy.get_parameter_descriptions()
            default_values = cls.get_default_parameters(strategy_type)

            # Build parameter schemas
            parameter_schemas = []

            # Add required parameters
            for param_name in required_params:
                param_schema = {
                    "name": param_name,
                    "type": cls._infer_parameter_type(
                        param_name, default_values.get(param_name)
                    ),
                    "description": param_descriptions.get(
                        param_name, f"Required parameter: {param_name}"
                    ),
                    "required": True,
                    "default_value": default_values.get(param_name),
                }
                parameter_schemas.append(param_schema)

            # Add optional parameters
            for param_name in optional_params:
                param_schema = {
                    "name": param_name,
                    "type": cls._infer_parameter_type(
                        param_name, default_values.get(param_name)
                    ),
                    "description": param_descriptions.get(
                        param_name, f"Optional parameter: {param_name}"
                    ),
                    "required": False,
                    "default_value": default_values.get(param_name),
                }
                parameter_schemas.append(param_schema)

            # Get example configurations
            examples = cls._get_strategy_examples(strategy_type)

            return {
                "strategy_name": strategy_name,
                "description": temp_strategy.get_description(),
                "parameters": parameter_schemas,
                "examples": examples,
            }

        except Exception as e:
            # Fallback implementation that doesn't require strategy instantiation
            default_values = cls.get_default_parameters(strategy_type)
            examples = cls._get_strategy_examples(strategy_type)

            # Build basic parameter schemas from defaults
            parameter_schemas = []
            for param_name, default_value in default_values.items():
                param_schema = {
                    "name": param_name,
                    "type": cls._infer_parameter_type(param_name, default_value),
                    "description": f"Parameter: {param_name}",
                    "required": False,
                    "default_value": default_value,
                }
                parameter_schemas.append(param_schema)

            return {
                "strategy_name": strategy_name,
                "description": f"Trading strategy: {strategy_name.replace('_', ' ').title()}",
                "parameters": parameter_schemas,
                "examples": examples,
            }

    @classmethod
    def _infer_parameter_type(cls, param_name: str, default_value: Any) -> str:
        """
        Infer parameter type from name and default value.

        Args:
            param_name: Parameter name
            default_value: Default value

        Returns:
            Inferred type as string
        """
        if default_value is not None:
            if isinstance(default_value, bool):
                return "boolean"
            elif isinstance(default_value, int):
                return "integer"
            elif isinstance(default_value, float):
                return "float"
            elif isinstance(default_value, str):
                return "string"
            elif isinstance(default_value, list):
                return "array"
            elif isinstance(default_value, dict):
                return "object"

        # Infer from parameter name patterns
        if "period" in param_name.lower() or "window" in param_name.lower():
            return "integer"
        elif (
            "threshold" in param_name.lower()
            or "ratio" in param_name.lower()
            or "factor" in param_name.lower()
        ):
            return "float"
        elif "size" in param_name.lower() and "position" in param_name.lower():
            return "float"
        elif "method" in param_name.lower() or "type" in param_name.lower():
            return "string"

        return "string"  # Default fallback

    @classmethod
    def _get_strategy_examples(cls, strategy_type: StrategyType) -> Dict[str, Any]:
        """
        Get example configurations for a strategy.

        Args:
            strategy_type: Strategy type

        Returns:
            Dictionary with example configurations
        """
        examples = {
            StrategyType.MOVING_AVERAGE_CROSSOVER: {
                "conservative": {
                    "fast_period": 10,
                    "slow_period": 30,
                    "position_sizing": "percentage",
                    "position_size": 0.5,
                },
                "aggressive": {
                    "fast_period": 5,
                    "slow_period": 15,
                    "position_sizing": "percentage",
                    "position_size": 0.8,
                },
            },
            StrategyType.RSI_STRATEGY: {
                "conservative": {
                    "rsi_period": 14,
                    "oversold_threshold": 25,
                    "overbought_threshold": 75,
                    "position_sizing": "percentage",
                    "position_size": 0.3,
                },
                "aggressive": {
                    "rsi_period": 10,
                    "oversold_threshold": 35,
                    "overbought_threshold": 65,
                    "position_sizing": "percentage",
                    "position_size": 0.7,
                },
            },
            StrategyType.BOLLINGER_BANDS: {
                "mean_reversion": {
                    "bb_period": 20,
                    "bb_std_dev": 2.0,
                    "bb_position": "mean_reversion",
                    "position_sizing": "percentage",
                    "position_size": 0.5,
                },
                "breakout": {
                    "bb_period": 15,
                    "bb_std_dev": 1.5,
                    "bb_position": "breakout",
                    "position_sizing": "percentage",
                    "position_size": 0.6,
                },
            },
            StrategyType.MACD_STRATEGY: {
                "standard": {
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "histogram_threshold": 0.0,
                    "position_sizing": "percentage",
                    "position_size": 0.5,
                },
                "sensitive": {
                    "fast_period": 8,
                    "slow_period": 21,
                    "signal_period": 6,
                    "histogram_threshold": 0.001,
                    "position_sizing": "percentage",
                    "position_size": 0.4,
                },
            },
            StrategyType.SIMPLE_BUY_HOLD: {
                "full_investment": {
                    "entry_delay": 0,
                    "position_sizing": "percentage",
                    "position_size": 1.0,
                },
                "partial_investment": {
                    "entry_delay": 0,
                    "position_sizing": "percentage",
                    "position_size": 0.8,
                },
            },
        }

        return examples.get(strategy_type, {})

    @classmethod
    def get_strategy_info(cls, strategy_name: str) -> Dict[str, Any]:
        """
        Get comprehensive information about a strategy.

        Args:
            strategy_name: Name of the strategy

        Returns:
            Dictionary containing strategy information

        Raises:
            StrategyError: If strategy is not available
        """
        if not cls.is_strategy_available(strategy_name):
            raise StrategyError(f"Strategy not available: {strategy_name}")

        strategy_type = StrategyType(strategy_name)
        strategy_class = cls._strategy_registry[strategy_type]

        # Create temporary instance to get information
        temp_config = StrategyConfig(strategy_type=strategy_type, parameters={})

        try:
            temp_strategy = strategy_class(temp_config)

            return {
                "name": strategy_name,
                "description": temp_strategy.get_description(),
                "required_parameters": temp_strategy.get_required_parameters(),
                "optional_parameters": temp_strategy.get_optional_parameters(),
                "parameter_descriptions": temp_strategy.get_parameter_descriptions(),
                "default_values": cls.get_default_parameters(strategy_type),
                "examples": cls._get_strategy_examples(strategy_type),
            }

        except Exception as e:
            # Return simplified info if the strategy can't be instantiated
            return {
                "name": strategy_name,
                "description": f"Trading strategy: {strategy_name}",
                "required_parameters": [],
                "optional_parameters": [],
                "parameter_descriptions": {},
                "default_values": cls.get_default_parameters(strategy_type),
                "examples": cls._get_strategy_examples(strategy_type),
            }

    @classmethod
    def get_all_strategies_info(cls) -> List[Dict[str, Any]]:
        """
        Get information for all available strategies.

        Returns:
            List of dictionaries containing strategy information
        """
        strategies_info = []

        for strategy_name in cls.get_available_strategies():
            try:
                strategy_info = cls.get_strategy_info(strategy_name)
                strategies_info.append(strategy_info)
            except Exception as e:
                # Log error but continue with other strategies
                print(f"Warning: Failed to get info for strategy {strategy_name}: {e}")
                continue

        return strategies_info

    @classmethod
    def validate_strategy_parameters(
        cls, strategy_name: str, parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate strategy parameters.

        Args:
            strategy_name: Name of the strategy
            parameters: Parameters to validate

        Returns:
            Dictionary with validation results

        Raises:
            StrategyError: If strategy is not available
        """
        if not cls.is_strategy_available(strategy_name):
            raise StrategyError(f"Strategy not available: {strategy_name}")

        strategy_type = StrategyType(strategy_name)
        strategy_class = cls._strategy_registry[strategy_type]

        # Create temporary instance to validate parameters
        temp_config = StrategyConfig(strategy_type=strategy_type, parameters=parameters)

        try:
            temp_strategy = strategy_class(temp_config)
            temp_strategy.validate_parameters()

            return {"valid": True, "errors": [], "warnings": []}

        except Exception as e:
            return {"valid": False, "errors": [str(e)], "warnings": []}
