"""Top-level package exports for the project.

Expose the main Model class and common training utilities at the package
level for convenience, while delegating implementations to subpackages.
"""

# Re-export the Model from the models subpackage
from .models import Model  # noqa: F401

# Re-export common training utilities from the utils subpackage
from .utils import (  # noqa: F401
    create_optimizer,
    seed_worker,
    set_seed,
    str_to_bool,
)

__all__ = [
    "Model",
    "create_optimizer",
    "seed_worker",
    "set_seed",
    "str_to_bool",
]
