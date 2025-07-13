from __future__ import annotations as _annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import date
from typing import Any
from xml.etree import ElementTree

from pydantic import BaseModel

__all__ = ("format_as_xml",)


def format_as_xml(
    obj: Any,
    root_tag: str = "examples",
    item_tag: str = "example",
    include_root_tag: bool = True,
    none_str: str = "null",
    indent: str | None = "  ",
) -> str:
    """Format a Python object as XML.

    This is useful since LLMs often find it easier to read semi-structured data (e.g. examples) as XML,
    rather than JSON etc.

    Supports: `str`, `bytes`, `bytearray`, `bool`, `int`, `float`, `date`, `datetime`, `Mapping`,
    `Iterable`, `dataclass`, and `BaseModel`.

    Args:
        obj: Python Object to serialize to XML.
        root_tag: Outer tag to wrap the XML in, use `None` to omit the outer tag.
        item_tag: Tag to use for each item in an iterable (e.g. list), this is overridden by the class name
            for dataclasses and Pydantic models.
        include_root_tag: Whether to include the root tag in the output
            (The root tag is always included if it includes a body - e.g. when the input is a simple value).
        none_str: String to use for `None` values.
        indent: Indentation string to use for pretty printing.

    Returns:
        XML representation of the object.

    Example:
    ```python {title="format_as_xml_example.py" lint="skip"}
    from pydantic_ai import format_as_xml

    print(format_as_xml({'name': 'John', 'height': 6, 'weight': 200}, root_tag='user'))
    '''
    <user>
      <name>John</name>
      <height>6</height>
      <weight>200</weight>
    </user>
    '''
    ```
    """
    to_xml_inst = _ToXml(item_tag=item_tag, none_str=none_str)
    el = to_xml_inst.to_xml(obj, root_tag)
    if not include_root_tag and el.text is None:
        join = "" if indent is None else "\n"
        return join.join(_rootless_xml_elements(el, indent))
    else:
        if indent is not None:
            ElementTree.indent(el, space=indent)
        return ElementTree.tostring(el, encoding="unicode")


@dataclass
class _ToXml:
    item_tag: str
    none_str: str

    def to_xml(self, value: Any, tag: str | None) -> ElementTree.Element:
        item_tag = self.item_tag
        none_str = self.none_str

        if tag is None:
            tag_val = item_tag
        else:
            tag_val = tag

        element = ElementTree.Element(tag_val)

        # Inline isinstance and shortcut for common/fast tests
        if value is None:
            element.text = none_str
            return element

        v_type = type(value)
        # Short-circuit string before Mapping/Iterable
        if v_type is str:
            element.text = value
            return element

        # Fast path for builtins
        if v_type in (int, float, bool):
            element.text = str(value)
            return element

        if isinstance(value, (bytes, bytearray)):
            # Only do decode if actually needed
            try:
                element.text = value.decode(errors="ignore")
            except Exception:
                element.text = ""
            return element

        if isinstance(value, date):
            element.text = value.isoformat()
            return element

        # Test for Mapping and Dataclass (but dataclass is also Mapping sometimes)
        # So: If is_dataclass check first, then Mapping for generic dict
        if is_dataclass(value) and not isinstance(value, type):
            # No deep asdict unless needed: Use value.__dict__ if possible for shallow and fast
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            if hasattr(value, "__dict__"):
                dc_dict = value.__dict__
            else:
                # fallback, slower
                dc_dict = asdict(value)
            self._mapping_to_xml(element, dc_dict)
            return element

        if isinstance(value, BaseModel):
            if tag is None:
                element = ElementTree.Element(value.__class__.__name__)
            # model_dump(mode='python') is faster than dict(model), avoid extra copy
            self._mapping_to_xml(element, value.model_dump(mode="python"))
            return element

        # Mapping must come after dataclass check
        if isinstance(value, Mapping):
            self._mapping_to_xml(element, value)
            return element

        # Fast path for common collections
        if isinstance(value, (list, tuple)):
            append = element.append
            for item in value:
                append(self.to_xml(item, None))
            return element

        # Generic slow Iterable
        if isinstance(value, Iterable) and not isinstance(
            value, (str, bytes, bytearray)
        ):
            append = element.append
            for item in value:
                append(self.to_xml(item, None))
            return element

        raise TypeError(f"Unsupported type for XML formatting: {type(value)}")

    def _mapping_to_xml(
        self, element: ElementTree.Element, mapping: Mapping[Any, Any]
    ) -> None:
        for key, value in mapping.items():
            if isinstance(key, int):
                key = str(key)
            elif not isinstance(key, str):
                raise TypeError(
                    f"Unsupported key type for XML formatting: {type(key)}, only str and int are allowed"
                )
            element.append(self.to_xml(value, key))


def _rootless_xml_elements(
    root: ElementTree.Element, indent: str | None
) -> Iterator[str]:
    if indent is not None:
        for sub_element in root:
            ElementTree.indent(sub_element, space=indent)
            yield ElementTree.tostring(sub_element, encoding="unicode")
    else:
        for sub_element in root:
            yield ElementTree.tostring(sub_element, encoding="unicode")
