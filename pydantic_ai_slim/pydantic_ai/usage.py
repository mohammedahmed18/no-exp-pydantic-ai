from __future__ import annotations as _annotations

from copy import copy
from dataclasses import dataclass

from . import _utils
from .exceptions import UsageLimitExceeded

__all__ = "Usage", "UsageLimits"


@dataclass(repr=False)
class Usage:
    """LLM usage associated with a request or run.

    Responsibility for calculating usage is on the model; PydanticAI simply sums the usage information across requests.

    You'll need to look up the documentation of the model you're using to convert usage to monetary costs.
    """

    requests: int = 0
    """Number of requests made to the LLM API."""
    request_tokens: int | None = None
    """Tokens used in processing requests."""
    response_tokens: int | None = None
    """Tokens used in generating responses."""
    total_tokens: int | None = None
    """Total tokens used in the whole run, should generally be equal to `request_tokens + response_tokens`."""
    details: dict[str, int] | None = None
    """Any extra details returned by the model."""

    def incr(self, incr_usage: Usage) -> None:
        """Increment the usage in place.

        Args:
            incr_usage: The usage to increment by.
        """
        # Manually access attributes to avoid getattr/setattr overhead
        # Also, variables are only assigned if necessary, reducing unnecessary writes

        s = self
        o = incr_usage
        sr = s.requests
        orr = o.requests
        if sr is not None or orr is not None:
            s.requests = (sr or 0) + (orr or 0)
        srt = s.request_tokens
        ort = o.request_tokens
        if srt is not None or ort is not None:
            s.request_tokens = (srt or 0) + (ort or 0)
        sresp = s.response_tokens
        oresp = o.response_tokens
        if sresp is not None or oresp is not None:
            s.response_tokens = (sresp or 0) + (oresp or 0)
        stt = s.total_tokens
        ott = o.total_tokens
        if stt is not None or ott is not None:
            s.total_tokens = (stt or 0) + (ott or 0)

        odetails = o.details
        if odetails:
            if s.details is None:
                s.details = odetails.copy()
            else:
                sd = s.details
                for k, v in odetails.items():
                    sd[k] = sd.get(k, 0) + v

    def __add__(self, other: Usage) -> Usage:
        """Add two Usages together.

        This is provided so it's trivial to sum usage information from multiple requests and runs.
        """
        # Manually shallow-copy only relevant fields for speed
        # Do not use copy() to avoid overhead for simple dataclasses
        s = self
        result = Usage.__new__(Usage)
        result.requests = s.requests
        result.request_tokens = s.request_tokens
        result.response_tokens = s.response_tokens
        result.total_tokens = s.total_tokens
        result.details = None if s.details is None else s.details.copy()
        result.incr(other)
        return result

    def opentelemetry_attributes(self) -> dict[str, int]:
        """Get the token limits as OpenTelemetry attributes."""
        result = {
            "gen_ai.usage.input_tokens": self.request_tokens,
            "gen_ai.usage.output_tokens": self.response_tokens,
        }
        for key, value in (self.details or {}).items():
            result[f"gen_ai.usage.details.{key}"] = value  # pragma: no cover
        return {k: v for k, v in result.items() if v}

    def has_values(self) -> bool:
        """Whether any values are set and non-zero."""
        return bool(
            self.requests or self.request_tokens or self.response_tokens or self.details
        )

    __repr__ = _utils.dataclasses_no_defaults_repr


@dataclass(repr=False)
class UsageLimits:
    """Limits on model usage.

    The request count is tracked by pydantic_ai, and the request limit is checked before each request to the model.
    Token counts are provided in responses from the model, and the token limits are checked after each response.

    Each of the limits can be set to `None` to disable that limit.
    """

    request_limit: int | None = 50
    """The maximum number of requests allowed to the model."""
    request_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests to the model."""
    response_tokens_limit: int | None = None
    """The maximum number of tokens allowed in responses from the model."""
    total_tokens_limit: int | None = None
    """The maximum number of tokens allowed in requests and responses combined."""

    def has_token_limits(self) -> bool:
        """Returns `True` if this instance places any limits on token counts.

        If this returns `False`, the `check_tokens` method will never raise an error.

        This is useful because if we have token limits, we need to check them after receiving each streamed message.
        If there are no limits, we can skip that processing in the streaming response iterator.
        """
        return any(
            limit is not None
            for limit in (
                self.request_tokens_limit,
                self.response_tokens_limit,
                self.total_tokens_limit,
            )
        )

    def check_before_request(self, usage: Usage) -> None:
        """Raises a `UsageLimitExceeded` exception if the next request would exceed the request_limit."""
        request_limit = self.request_limit
        if request_limit is not None and usage.requests >= request_limit:
            raise UsageLimitExceeded(
                f"The next request would exceed the request_limit of {request_limit}"
            )

    def check_tokens(self, usage: Usage) -> None:
        """Raises a `UsageLimitExceeded` exception if the usage exceeds any of the token limits."""
        request_tokens = usage.request_tokens or 0
        if (
            self.request_tokens_limit is not None
            and request_tokens > self.request_tokens_limit
        ):
            raise UsageLimitExceeded(
                f"Exceeded the request_tokens_limit of {self.request_tokens_limit} ({request_tokens=})"
            )

        response_tokens = usage.response_tokens or 0
        if (
            self.response_tokens_limit is not None
            and response_tokens > self.response_tokens_limit
        ):
            raise UsageLimitExceeded(
                f"Exceeded the response_tokens_limit of {self.response_tokens_limit} ({response_tokens=})"
            )

        total_tokens = usage.total_tokens or 0
        if (
            self.total_tokens_limit is not None
            and total_tokens > self.total_tokens_limit
        ):
            raise UsageLimitExceeded(
                f"Exceeded the total_tokens_limit of {self.total_tokens_limit} ({total_tokens=})"
            )

    __repr__ = _utils.dataclasses_no_defaults_repr
