from typing import Any, Protocol

from pydantic.json_schema import JsonSchemaValue

from pydantic_ai.tools import Tool as BaseTool, Tool


class LangChainTool(Protocol):
    # args are like
    # {'dir_path': {'default': '.', 'description': 'Subdirectory to search in.', 'title': 'Dir Path', 'type': 'string'},
    #  'pattern': {'description': 'Unix shell regex, where * matches everything.', 'title': 'Pattern', 'type': 'string'}}
    @property
    def args(self) -> dict[str, JsonSchemaValue]: ...

    def get_input_jsonschema(self) -> JsonSchemaValue: ...

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str: ...

    def run(self, *args: Any, **kwargs: Any) -> str: ...


__all__ = ("tool_from_langchain",)


def tool_from_langchain(langchain_tool: LangChainTool) -> Tool:
    """Creates a Pydantic AI tool proxy from a LangChain tool.

    Args:
        langchain_tool: The LangChain tool to wrap.

    Returns:
        A Pydantic AI tool that corresponds to the LangChain tool.
    """
    # Localize attribute and method lookups for speed
    name = langchain_tool.name
    description = langchain_tool.description
    args_dict = langchain_tool.args
    get_input_jsonschema = langchain_tool.get_input_jsonschema

    schema = get_input_jsonschema()
    # Use direct assignment and only set required if there are any
    # Avoid repeated obj lookups and dict comprehensions
    required_list = []
    defaults = {}
    for key, detail in args_dict.items():
        if "default" in detail:
            defaults[key] = detail["default"]
        else:
            required_list.append(key)
    if "additionalProperties" not in schema:
        schema["additionalProperties"] = False
    if required_list:
        schema["required"] = sorted(required_list)

    def proxy(*args: Any, **kwargs: Any) -> str:
        # This function should only be called with kwargs
        assert not args, "This should always be called with kwargs"
        if defaults:
            # Create a merged dict without copying unless needed
            if not kwargs:
                merged_kwargs = defaults
            elif not defaults:
                merged_kwargs = kwargs
            else:
                merged_kwargs = defaults.copy()
                merged_kwargs.update(kwargs)
            return langchain_tool.run(merged_kwargs)
        else:
            return langchain_tool.run(kwargs)

    return BaseTool.from_schema(
        function=proxy,
        name=name,
        description=description,
        json_schema=schema,
    )
