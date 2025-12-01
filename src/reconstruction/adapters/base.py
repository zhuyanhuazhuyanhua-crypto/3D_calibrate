from abc import ABC, abstractmethod
from typing import Optional, Dict


class AdapterBase(ABC):
    """Abstract adapter interface for external tools (SfM/MVS/Converter).

    Implementations should be lightweight wrappers that translate our pipeline calls
    into concrete binary invocations or library calls.
    """

    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError()

    def is_available(self) -> bool:
        """Return True if the underlying tool is available on this system."""
        return False

    def info(self) -> Dict:
        return {'name': self.name(), 'available': self.is_available()}
