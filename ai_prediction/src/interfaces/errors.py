# interfaces/errors.py


class CollectionError(Exception):
    """Raised when data collection fails"""

    pass


class ValidationError(Exception):
    """Raised when input validation fails"""

    pass


class RepositoryError(Exception):
    """Raised when repository operations fail"""

    pass
