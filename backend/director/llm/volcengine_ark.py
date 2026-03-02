import os
import json
from pydantic import Field, field_validator, FieldValidationInfo
from pydantic_settings import SettingsConfigDict

from director.llm.base import BaseLLM, BaseLLMConfig, LLMResponse, LLMResponseStatus
from director.constants import LLMType, EnvPrefix

class VolcengineConfig(BaseLLMConfig):
    model_config = SettingsConfigDict(
        env_prefix=EnvPrefix.VOLCENGINE_,
        extra="ignore",
    )

    llm_type: str = LLMType.VOLCENGINE
    api_key: str = ""
    api_base: str = "https://ark.cn-beijing.volces.com/api/v3"
    chat_model: str = Field(default="ep-20260219195713-2nbl9")
    max_tokens: int = 4096

    @field_validator("api_key")
    @classmethod
    def validate_non_empty(cls, v, info: FieldValidationInfo):
        if not v:
            raise ValueError(f"{info.field_name} must not be empty. please set ARK_API_KEY environment variable.")
        return v

class VolcengineArkAI(BaseLLM):
    def __init__(self, config: VolcengineConfig = None):
        if config is None:
            config = VolcengineConfig(api_key=os.environ.get("ARK_API_KEY", ""))
        super().__init__(config=config)
        try:
            from volcenginesdkarkruntime import Ark
        except ImportError:
            raise ImportError("Please run: pip install volcenginesdkarkruntime")
        
        self.client = Ark(api_key=self.api_key, base_url=self.api_base)
        
    def _format_messages(self, messages: list):
        formatted_messages = []
        for message in messages:
            if message["role"] == "assistant" and message.get("tool_calls"):
                formatted_messages.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                        "tool_calls": [
                            {
                                "id": tool_call["id"],
                                "function": {
                                    "name": tool_call["tool"]["name"],
                                    "arguments": json.dumps(
                                        tool_call["tool"]["arguments"]
                                    ),
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
            formatted_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"],
                }
            })
        return formatted_tools

    def chat_completions(self, messages: list, tools: list = [], stop=None, response_format=None):
        params = {
            "model": self.chat_model,
            "messages": self._format_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "timeout": self.timeout,
        }
        if tools:
            params["tools"] = self._format_tools(tools)
            params["tool_choice"] = "auto"
            
        if response_format:
            params["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            return LLMResponse(content=f"Error: {str(e)}", status=LLMResponseStatus.ERROR)

        choice = response.choices[0]
        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=[
                {
                    "id": tc.id,
                    "tool": {
                        "name": tc.function.name,
                        "arguments": json.loads(tc.function.arguments),
                    },
                    "type": tc.type,
                }
                for tc in getattr(choice.message, "tool_calls", []) or []
            ] if getattr(choice.message, "tool_calls", []) else [],
            finish_reason=choice.finish_reason,
            send_tokens=response.usage.prompt_tokens,
            recv_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            status=LLMResponseStatus.SUCCESS,
        )
