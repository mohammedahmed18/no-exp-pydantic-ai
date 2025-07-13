from __future__ import annotations as _annotations

import warnings

from pydantic_ai.exceptions import UserError

from . import ModelProfile
from ._json_schema import JsonSchema, JsonSchemaTransformer


def google_model_profile(model_name: str) -> ModelProfile | None:
    """Get the model profile for a Google model."""
    return ModelProfile(
        json_schema_transformer=GoogleJsonSchemaTransformer,
        supports_json_schema_output=True,
        supports_json_object_output=True,
    )


class GoogleJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema from Pydantic to be suitable for Gemini.

    Gemini which [supports](https://ai.google.dev/gemini-api/docs/function-calling#function_declarations)
    a subset of OpenAPI v3.0.3.

    Specifically:
    * gemini doesn't allow the `title` keyword to be set
    * gemini doesn't allow `$defs` — we need to inline the definitions where possible
    """

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True, simplify_nullable_unions=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        # Fast path for 'additionalProperties'
        if 'additionalProperties' in schema:
            additional_properties = schema.pop('additionalProperties')
            if additional_properties:
                original_schema = {**schema, 'additionalProperties': additional_properties}
                warnings.warn(
                    '`additionalProperties` is not supported by Gemini; it will be removed from the tool JSON schema.'
                    f' Full schema: {self.schema}\n\n'
                    f'Source of additionalProperties within the full schema: {original_schema}\n\n'
                    'If this came from a field with a type like `dict[str, MyType]`, that field will always be empty.\n\n'
                    "If Google's APIs are updated to support this properly, please create an issue on the PydanticAI GitHub"
                    ' and we will fix this behavior.',
                    UserWarning,
                )

        schema.pop('title', None)
        schema.pop('default', None)
        schema.pop('$schema', None)
        const = schema.pop('const', None)
        if const is not None:
            # Gemini doesn't support const, but it does support enum with a single value
            schema['enum'] = [const]
        schema.pop('discriminator', None)
        schema.pop('examples', None)
        schema.pop('exclusiveMaximum', None)
        schema.pop('exclusiveMinimum', None)

        # Fast path for stringifying enum values
        enum = schema.get('enum')
        if enum:
            schema['type'] = 'string'
            # Use set to avoid repeated conversions to string for exact same object
            str_enum = [str(val) for val in enum]
            schema['enum'] = str_enum

        type_ = schema.get('type')
        if 'oneOf' in schema and 'type' not in schema:  # pragma: no cover
            schema['anyOf'] = schema.pop('oneOf')

        if type_ == 'string':
            fmt = schema.pop('format', None)
            if fmt is not None:
                description = schema.get('description')
                if description:
                    schema['description'] = f'{description} (format: {fmt})'
                else:
                    schema['description'] = f'Format: {fmt}'

        if '$ref' in schema:
            raise UserError(f'Recursive `$ref`s in JSON Schema are not supported by Gemini: {schema["$ref"]}')

        if 'prefixItems' in schema:
            prefix_items = schema.pop('prefixItems')
            items = schema.get('items')
            if items is not None:
                # Deduplicate by object id for both correctness and performance (JSON Schemas are dicts, so id is good)
                unique_items = [items]
                seen_ids = {id(items)}
            else:
                unique_items = []
                seen_ids = set()
            for item in prefix_items:
                item_id = id(item)
                if item_id not in seen_ids:
                    unique_items.append(item)
                    seen_ids.add(item_id)
            n_unique = len(unique_items)
            if n_unique > 1:  # pragma: no cover
                schema['items'] = {'anyOf': unique_items}
            elif n_unique == 1:  # pragma: no branch
                schema['items'] = unique_items[0]
            schema.setdefault('minItems', len(prefix_items))
            if items is None:  # pragma: no branch
                schema.setdefault('maxItems', len(prefix_items))

        return schema
