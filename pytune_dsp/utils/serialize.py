# pytune_dsp/utils/serialize.py
from dataclasses import is_dataclass, asdict

def safe_asdict(obj):
    """
    Sérialisation robuste : dataclass, Pydantic ou dict.
    Évite le warning "asdict() should be called on dataclass instances".
    """
    if obj is None:
        return None
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):  # Pydantic v2
        return obj.model_dump()
    if hasattr(obj, "dict"):  # Pydantic v1
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    try:
        return {k: safe_asdict(v) for k, v in obj.__dict__.items()}
    except Exception:
        return {"value": str(obj)}