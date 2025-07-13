from __future__ import annotations as _annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

from pydantic_ai.exceptions import UserError

JsonSchema = dict[str, Any]


@dataclass(init=False)
class JsonSchemaTransformer(ABC):
    """Walks a JSON schema, applying transformations to it at each level.

    Note: We may eventually want to rework tools to build the JSON schema from the type directly, using a subclass of
    pydantic.json_schema.GenerateJsonSchema, rather than making use of this machinery.
    """

    def __init__(
        self,
        schema: 'JsonSchema',
        *,
        strict: bool | None = None,
        prefer_inlined_defs: bool = False,
        simplify_nullable_unions: bool = False,
    ):
        self.schema = schema
        self.strict = strict
        self.is_strict_compatible = True  # Can be set to False by subclasses to set `strict` on `ToolDefinition` when set not set by user explicitly
        self.prefer_inlined_defs = prefer_inlined_defs
        self.simplify_nullable_unions = simplify_nullable_unions

        self.defs: dict[str, 'JsonSchema'] = self.schema.get('$defs', {})
        self.refs_stack: list[str] = []
        self.recursive_refs = set[str]()

    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema:
        """Make changes to the schema."""
        return schema

    def walk(self) -> JsonSchema:
        # Optimization 1: Avoid expensive deepcopy; only shallow copy and copy $defs on demand.
        # It's safe to copy only the dict itself since we manage $defs separately and _handle makes fresh copies as needed.
        schema = dict(self.schema)
        raw_defs = schema.pop('$defs', None)

        # Optimization 2: Fast-path if $defs is not present or empty (majority of cases).
        if raw_defs is not None:
            self.defs = raw_defs

        handled = self._handle(schema)

        # Only post-process $defs if required
        if not self.prefer_inlined_defs and self.defs:
            # Use generator version for dict comprehension to reduce memory pressure, call inline
            defs_handled = {}
            for k, v in self.defs.items():
                defs_handled[k] = self._handle(v)
            handled['$defs'] = defs_handled

        elif self.recursive_refs:  # pragma: no cover
            defs = {key: self.defs[key] for key in self.recursive_refs}
            root_ref = self.schema.get('$ref')
            root_key = None if root_ref is None else remove_defs_prefix(root_ref)  # uses helper instead of regex
            if root_key is None:
                root_key = self.schema.get('title', 'root')
                while root_key in defs:
                    # Modify the root key until it is not already in use
                    root_key = f'{root_key}_root'

            defs[root_key] = handled
            return {'$defs': defs, '$ref': f'#/$defs/{root_key}'}

        return handled

    def _handle(self, schema: JsonSchema) -> JsonSchema:
        nested_refs = 0
        if self.prefer_inlined_defs:
            while True:
                ref = schema.get('$ref')
                if not ref:
                    break
                # Optimization: replace expensive regex with a direct string method
                key = remove_defs_prefix(ref)
                if key in self.refs_stack:
                    self.recursive_refs.add(key)
                    break  # recursive ref can't be unpacked
                self.refs_stack.append(key)
                nested_refs += 1

                def_schema = self.defs.get(key)
                if def_schema is None:  # pragma: no cover
                    raise UserError(f'Could not find $ref definition for {key}')
                schema = def_schema

        # Handle the schema based on its type / structure
        type_ = schema.get('type')
        if type_ == 'object':
            schema = self._handle_object(schema)
        elif type_ == 'array':
            schema = self._handle_array(schema)
        elif type_ is None:
            schema = self._handle_union(schema, 'anyOf')
            schema = self._handle_union(schema, 'oneOf')

        schema = self.transform(schema)

        if nested_refs > 0:
            del self.refs_stack[-nested_refs:]  # Use del for efficient slice deletion

        return schema

    def _handle_object(self, schema: JsonSchema) -> JsonSchema:
        if properties := schema.get('properties'):
            handled_properties = {}
            for key, value in properties.items():
                handled_properties[key] = self._handle(value)
            schema['properties'] = handled_properties

        if (additional_properties := schema.get('additionalProperties')) is not None:
            if isinstance(additional_properties, bool):
                schema['additionalProperties'] = additional_properties
            else:
                schema['additionalProperties'] = self._handle(additional_properties)

        if (pattern_properties := schema.get('patternProperties')) is not None:
            handled_pattern_properties = {}
            for key, value in pattern_properties.items():
                handled_pattern_properties[key] = self._handle(value)
            schema['patternProperties'] = handled_pattern_properties

        return schema

    def _handle_array(self, schema: JsonSchema) -> JsonSchema:
        if prefix_items := schema.get('prefixItems'):
            schema['prefixItems'] = [self._handle(item) for item in prefix_items]

        if items := schema.get('items'):
            schema['items'] = self._handle(items)

        return schema

    def _handle_union(self, schema: JsonSchema, union_kind: Literal['anyOf', 'oneOf']) -> JsonSchema:
        members = schema.get(union_kind)
        if not members:
            return schema

        handled = [self._handle(member) for member in members]

        # convert nullable unions to nullable types
        if self.simplify_nullable_unions:
            handled = self._simplify_nullable_union(handled)

        if len(handled) == 1:
            # In this case, no need to retain the union
            return handled[0]

        # If we have keys besides the union kind (such as title or discriminator), keep them without modifications
        schema = schema.copy()
        schema[union_kind] = handled
        return schema

    @staticmethod
    def _simplify_nullable_union(cases: list[JsonSchema]) -> list[JsonSchema]:
        # TODO: Should we move this to relevant subclasses? Or is it worth keeping here to make reuse easier?
        if len(cases) == 2 and {'type': 'null'} in cases:
            # Find the non-null schema
            non_null_schema = next(
                (item for item in cases if item != {'type': 'null'}),
                None,
            )
            if non_null_schema:
                # Create a new schema based on the non-null part, mark as nullable
                new_schema = deepcopy(non_null_schema)
                new_schema['nullable'] = True
                return [new_schema]
            else:  # pragma: no cover
                # they are both null, so just return one of them
                return [cases[0]]

        return cases


class InlineDefsJsonSchemaTransformer(JsonSchemaTransformer):
    """Transforms the JSON Schema to inline $defs."""

    def __init__(self, schema: JsonSchema, *, strict: bool | None = None):
        super().__init__(schema, strict=strict, prefer_inlined_defs=True)

    def transform(self, schema: JsonSchema) -> JsonSchema:
        return schema


# Use removeprefix if available for regex replacement optimization (Python 3.9+)
def remove_defs_prefix(s):
    prefix = '#/$defs/'
    if s.startswith(prefix):
        return s[len(prefix):]
    return s
