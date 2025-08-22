"""A registry for attack classes."""

ATTACK_REGISTRY = {}

def register_attack(name: str):
    """A decorator to register a new attack class."""
    def decorator(cls):
        if name in ATTACK_REGISTRY:
            raise ValueError(f"Attack {name} already registered.")
        ATTACK_REGISTRY[name] = cls
        return cls
    return decorator
