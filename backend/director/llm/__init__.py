import os

from director.constants import LLMType

from director.llm.openai import OpenAI
from director.llm.anthropic import AnthropicAI
from director.llm.googleai import GoogleAI
from director.llm.ark import ArkLLM, _load_from_llm_env
from director.llm.videodb_proxy import VideoDBProxy


def get_default_llm():
    """Get default LLM"""

    openai = True if os.getenv("OPENAI_API_KEY") else False
    anthropic = True if os.getenv("ANTHROPIC_API_KEY") else False
    googleai = True if os.getenv("GOOGLEAI_API_KEY") else False
    ark = True if os.getenv("ARK_API_KEY") else False
    if not ark:
        fallback_key, _ = _load_from_llm_env()
        ark = True if fallback_key else False

    default_llm = os.getenv("DEFAULT_LLM")

    if default_llm == LLMType.ARK:
        return ArkLLM()
    if default_llm == LLMType.OPENAI:
        return OpenAI()
    if default_llm == LLMType.ANTHROPIC:
        return AnthropicAI()
    if default_llm == LLMType.GOOGLEAI:
        return GoogleAI()

    if ark:
        return ArkLLM()
    if openai:
        return OpenAI()
    if anthropic:
        return AnthropicAI()
    if googleai:
        return GoogleAI()
    return VideoDBProxy()
