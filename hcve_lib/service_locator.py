class DependencyInjectionService:
    _services = {}
    current_context = {}

    @classmethod
    def register(cls, name, service):
        cls._services[name] = service

    @classmethod
    def get(cls, name):
        # Check the current context first for overridden services
        service = cls.current_context.get(name)
        if service is not None:
            return service

        # Return the global service if not overridden
        return cls._services.get(name)

    @classmethod
    def inject(cls, services):
        """Returns a context manager that temporarily injects services."""
        return _ServiceInjector(services)


class _ServiceInjector:
    def __init__(self, services):
        self.services = services

    def __enter__(self):
        # Store the services in the current context
        DependencyInjectionService.current_context.update(self.services)

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove the services from the current context
        for name in self.services:
            DependencyInjectionService.current_context.pop(name, None)


di = DependencyInjectionService()
