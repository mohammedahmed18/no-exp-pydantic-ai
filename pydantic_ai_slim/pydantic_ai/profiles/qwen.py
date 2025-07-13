from __future__ import annotations as _annotations

from . import ModelProfile
from ._json_schema import InlineDefsJsonSchemaTransformer


def qwen_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Qwen model."""
    return _model_profile_instance


_inline_defs_json_schema_transformer = InlineDefsJsonSchemaTransformer

_model_profile_instance = ModelProfile(
    json_schema_transformer=_inline_defs_json_schema_transformer
)
