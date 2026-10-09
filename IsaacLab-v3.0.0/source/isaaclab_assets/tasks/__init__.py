"""Custom task implementations for isaaclab_assets."""

from .my_exoskeleton_flat import MyExoskeletonFlatEnv

__all__ = ["MyExoskeletonFlatEnv", "MyExoskeletonImitationEnv"]


def __getattr__(name: str):
    """Load the imitation environment only after its config module is ready.

    The imitation task imports its environment configuration, which in turn
    imports this package for shared task helpers.  Eagerly importing both from
    this module creates a cycle during config discovery; a lazy attribute keeps
    the public ``isaaclab_assets.tasks.MyExoskeletonImitationEnv`` API intact.
    """

    if name == "MyExoskeletonImitationEnv":
        from .my_exoskeleton_imitation import MyExoskeletonImitationEnv

        return MyExoskeletonImitationEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
