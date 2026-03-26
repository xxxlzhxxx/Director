import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import LLMType, EnvPrefix


def _load_from_llm_env(md_path: str = "/Users/bytedance/Desktop/Workspace/code/LLM_env.md") -> Tuple[Optional[str], Optional[str]]:
    path = Path(md_path)
    if not path.exists():
        return None, None
    text = path.read_text(encoding="utf-8")
    api_key = None
    chat_model = None
    m_key = re.search(r"API Key:\s*([^\s]+)", text)
    if m_key:
        api_key = m_key.group(1)
    m_chat = re.search(r"doubao-seed-2-0-pro[^\n]*\nEndpoint:\s*(ep-[\w-]+)", text)
    if m_chat:
        chat_model = m_chat.group(1)
    return api_key, chat_model


class ArkConfig(BaseLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix=EnvPrefix.ARK_,
        extra="ignore",
    )

    llm_type: str = LLMType.ARK
    api_key: str = ""
    api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    chat_model: str = Field(default="")
    max_tokens: int = 4096


class ArkLLM(BaseLLM):
    def __init__(self, config: ArkConfig = None):
        if config is None:
            config = ArkConfig()

        if not config.api_key or not config.chat_model:
            fallback_key, fallback_chat_model = _load_from_llm_env()
            if not config.api_key and fallback_key:
                config.api_key = fallback_key
            if not config.chat_model and fallback_chat_model:
                config.chat_model = fallback_chat_model

        if not config.chat_model:
            config.chat_model = os.getenv("ARK_CHAT_MODEL") or os.getenv("ARK_TEXT_MODEL") or ""

        if not config.api_key:
            raise ValueError(
                f"ARK api_key is required. Set {EnvPrefix.ARK_.value}API_KEY or provide it in LLM_env.md."
            )
        if not config.chat_model:
            raise ValueError(
                f"ARK chat_model is required. Set {EnvPrefix.ARK_.value}CHAT_MODEL or provide it in LLM_env.md."
            )

        super().__init__(config=config)
        try:
            import openai
        except ImportError:
            raise ImportError("Please install OpenAI python library.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )

    def _format_messages(self, messages: list):
        formatted_messages = []
        for message in messages:
            if message.get("tool_calls"):
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": message.get("content", ""),
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "function": {
                                    "name": tool_call["tool"]["name"],
                                    "arguments": json.dumps(tool_call["tool"]["arguments"]),
                                },
                                "type": tool_call["type"],
                            }
                            for tool_call in message["tool_calls"]
                        ],
                    }
                )
            else:
                formatted_messages.append(message)
        return formatted_messages

    def _format_tools(self, tools: list):
        formatted_tools = []
        for tool in tools:
            formatted_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                    "strict": True,
                }
            )
        return formatted_tools

    def chat_completions(self, messages: List[Dict], tools: List[Dict] = None) -> LLMResponse:
        try:
            kwargs = {
                "model": self.chat_model,
                "messages": self._format_messages(messages),
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
            }
            if tools:
                kwargs["tools"] = self._format_tools(tools)
                kwargs["tool_choice"] = "auto"
            response = self.client.chat.completions.create(**kwargs)
            return LLMResponse(
                content=response.choices[0].message.content or "",
                tool_calls=[
                    {
                        "id": tool_call.id,
                        "tool": {
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments),
                        },
                        "type": tool_call.type,
                    }
                    for tool_call in response.choices[0].message.tool_calls
                ]
                if response.choices[0].message.tool_calls
                else [],
                finish_reason=response.choices[0].finish_reason,
                send_tokens=response.usage.prompt_tokens,
                recv_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                status=LLMResponseStatus.SUCCESS,
            )
        except Exception as e:
            return LLMResponse(content=str(e), status=LLMResponseStatus.ERROR)
