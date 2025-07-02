from typing import Optional, Dict, Any
from dependency_injector import containers, providers
from pydantic_settings import BaseSettings

from .llm_interfaces import LLMInterface
from .llm_factory import LLMFactory, LLMProvider, StrategyType
from .llm_facade import LLMFacade
from ..logger import LoggerFactory, LoggerType, LogLevel

logger = LoggerFactory.get_logger(
    name="llm-di-container", logger_type=LoggerType.STANDARD, level=LogLevel.INFO
)


class LLMSettings(BaseSettings):
    """LLM configuration settings"""

    provider: str = "openai"
    strategy: str = "simple"
    default_model: str = "gpt-4o-mini"
    api_key: Optional[str] = None

    class Config:
        env_prefix = "LLM_"


class LLMContainer(containers.DeclarativeContainer):
    """Dependency injection container for LLM services"""

    # Configuration
    config = providers.Configuration()

    # Services
    llm_factory = providers.Singleton(LLMFactory)

    llm_interface = providers.Factory(
        LLMFactory.get_llm,
        provider=config.provider,
        strategy=config.strategy,
        default_model=config.default_model,
        api_key=config.api_key,
    )

    llm_facade = providers.Factory(
        LLMFacade,
        provider=config.provider,
        strategy=config.strategy,
        default_model=config.default_model,
        api_key=config.api_key,
    )


# Global container instance
_container: Optional[LLMContainer] = None


def get_container() -> LLMContainer:
    """Get global DI container instance"""
    global _container
    if _container is None:
        _container = LLMContainer()
        # Load default configuration
        settings = LLMSettings()
        _container.config.from_dict(
            {
                "provider": LLMProvider(settings.provider),
                "strategy": StrategyType(settings.strategy),
                "default_model": settings.default_model,
                "api_key": settings.api_key,
            }
        )
        logger.info("LLM DI container initialized")
    return _container


def configure_container(
    provider: LLMProvider = LLMProvider.OPENAI,
    strategy: StrategyType = StrategyType.SIMPLE,
    model: str = "gpt-4o-mini",
    **kwargs,
) -> LLMContainer:
    """Configure global DI container"""
    container = get_container()
    container.config.from_dict(
        {"provider": provider, "strategy": strategy, "default_model": model, **kwargs}
    )
    logger.info(f"Container configured: {provider.value} + {strategy.value}")
    return container


# Convenience functions
def get_llm() -> LLMInterface:
    """Get LLM interface from global container"""
    return get_container().llm_interface()


def get_facade() -> LLMFacade:
    """Get LLM facade from global container"""
    return get_container().llm_facade()

    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[[], T],
        scope: Scope = Scope.SINGLETON,
    ) -> None:
        """
        Register a service with factory function

        Args:
            service_type: Service interface type
            factory: Factory function to create instances
            scope: Service scope
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type, factory=factory, scope=scope
        )
        logger.debug(
            f"Registered factory: {service_type.__name__} (scope: {scope.value})"
        )

    def register_instance(self, service_type: Type[T], instance: T) -> None:
        """
        Register a specific instance

        Args:
            service_type: Service interface type
            instance: Service instance
        """
        self._services[service_type] = ServiceDescriptor(
            service_type=service_type,
            instance=instance,
            scope=Scope.SINGLETON,
            configured=True,
        )
        self._instances[service_type] = instance
        logger.debug(f"Registered instance: {service_type.__name__}")

    def resolve(self, service_type: Type[T]) -> T:
        """
        Resolve a service instance

        Args:
            service_type: Service type to resolve

        Returns:
            Service instance

        Raises:
            ValueError: If service not registered
        """
        if service_type not in self._services:
            raise ValueError(f"Service {service_type.__name__} not registered")

        descriptor = self._services[service_type]

        # Return existing instance if singleton and already created
        if descriptor.scope == Scope.SINGLETON and service_type in self._instances:
            logger.debug(f"Returning existing singleton: {service_type.__name__}")
            return self._instances[service_type]

        # Create new instance
        instance = self._create_instance(descriptor)

        # Cache singleton instances
        if descriptor.scope == Scope.SINGLETON:
            self._instances[service_type] = instance
            logger.debug(f"Created and cached singleton: {service_type.__name__}")
        else:
            logger.debug(f"Created transient instance: {service_type.__name__}")

        return instance

    def _create_instance(self, descriptor: ServiceDescriptor) -> Any:
        """Create service instance from descriptor"""
        if descriptor.instance is not None:
            return descriptor.instance

        if descriptor.factory is not None:
            try:
                return descriptor.factory()
            except Exception as e:
                logger.error(
                    f"Factory failed for {descriptor.service_type.__name__}: {e}"
                )
                raise

        if descriptor.implementation is not None:
            try:
                return descriptor.implementation()
            except Exception as e:
                logger.error(
                    f"Implementation instantiation failed for {descriptor.service_type.__name__}: {e}"
                )
                raise

        raise ValueError(
            f"No way to create instance for {descriptor.service_type.__name__}"
        )

    def configure_llm(
        self,
        provider: LLMProvider = LLMProvider.OPENAI,
        strategy: StrategyType = StrategyType.SIMPLE,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> None:
        """
        Configure default LLM services

        Args:
            provider: LLM provider
            strategy: Generation strategy
            model: Default model
            **kwargs: Additional configuration
        """
        logger.info(f"Configuring LLM services: {provider.value} + {strategy.value}")

        # Register configured LLM interface
        self.register_factory(
            LLMInterface,
            lambda: LLMFactory.get_llm(
                provider=provider, strategy=strategy, default_model=model, **kwargs
            ),
        )

        # Register configured LLM facade
        self.register_factory(
            LLMFacade,
            lambda: LLMFacade(
                provider=provider, strategy=strategy, default_model=model, **kwargs
            ),
        )

        logger.info("LLM services configured successfully")

    def configure_openai_llm(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        strategy: StrategyType = StrategyType.SIMPLE,
        **kwargs,
    ) -> None:
        """
        Configure OpenAI LLM services

        Args:
            api_key: OpenAI API key
            model: OpenAI model
            strategy: Generation strategy
            **kwargs: Additional configuration
        """
        self.configure_llm(
            provider=LLMProvider.OPENAI,
            strategy=strategy,
            model=model,
            api_key=api_key,
            **kwargs,
        )

    def get_llm(self) -> LLMInterface:
        """Get configured LLM interface"""
        return self.resolve(LLMInterface)

    def get_facade(self) -> LLMFacade:
        """Get configured LLM facade"""
        return self.resolve(LLMFacade)

    def clear_singletons(self) -> None:
        """Clear all singleton instances"""
        logger.info(f"Clearing {len(self._instances)} singleton instances")
        self._instances.clear()

    def get_registered_services(self) -> Dict[str, Dict[str, Any]]:
        """Get information about registered services"""
        return {
            service_type.__name__: {
                "scope": descriptor.scope.value,
                "has_implementation": descriptor.implementation is not None,
                "has_factory": descriptor.factory is not None,
                "has_instance": descriptor.instance is not None,
                "configured": descriptor.configured,
                "is_instantiated": service_type in self._instances,
            }
            for service_type, descriptor in self._services.items()
        }

    def health_check(self) -> Dict[str, bool]:
        """Perform health check on registered services"""
        results = {}

        try:
            llm = self.get_llm()
            results["llm_interface"] = llm.is_available()
        except Exception as e:
            logger.error(f"LLM interface health check failed: {e}")
            results["llm_interface"] = False

        try:
            facade = self.get_facade()
            results["llm_facade"] = facade.is_healthy()
        except Exception as e:
            logger.error(f"LLM facade health check failed: {e}")
            results["llm_facade"] = False

        return results
