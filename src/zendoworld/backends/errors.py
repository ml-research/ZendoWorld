"""Errors shared by backend clients and experiment orchestration."""


class OpenAIQuotaExhausted(BaseException):
    """Abort the complete experiment when the API account has no quota."""
