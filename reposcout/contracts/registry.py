from collections.abc import Callable

from .localizer import Localizer

type LocalizerFactory = Callable[[], Localizer]
LOCALIZERS: dict[str, LocalizerFactory] = {}


def register[F: LocalizerFactory](name: str) -> Callable[[F], F]:
    def decorate(factory: F) -> F:
        if not name.strip() or name in LOCALIZERS:
            raise ValueError(f"empty or duplicate localizer name: {name!r}")
        LOCALIZERS[name] = factory
        return factory

    return decorate
