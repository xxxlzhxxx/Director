import logging
import os
import uuid

from abc import ABC, abstractmethod
from pydantic import BaseModel

from openai_function_calling import FunctionInferrer

from director.core.session import Session, OutputMessage
from director.constants import RUNS_PATH

logger = logging.getLogger(__name__)


class AgentStatus:
    SUCCESS = "success"
    ERROR = "error"


class AgentResponse(BaseModel):
    """Data model for respones from agents."""

    status: str = AgentStatus.SUCCESS
    message: str = ""
    data: dict = {}


class BaseAgent(ABC):
    """Interface for all agents. All agents should inherit from this class."""

    def __init__(self, session: Session, **kwargs):
        self.session: Session = session
        self.output_message: OutputMessage = self.session.output_message
        self._ensure_run_dir()

    def _ensure_run_dir(self) -> str:
        run_dir = self.session.state.get("run_dir")
        if isinstance(run_dir, str) and run_dir:
            return run_dir

        session_id = self.session.session_id or "session"
        conv_id = self.session.conv_id or "conv"
        msg_id = getattr(self.session.output_message, "msg_id", "") or "msg"
        suffix = uuid.uuid4().hex[:8]
        run_id = f"{session_id}_{conv_id}_{msg_id}_{suffix}"
        run_id = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in run_id)

        run_dir = os.path.join(RUNS_PATH, run_id)
        os.makedirs(os.path.join(run_dir, "script"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "assets"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "results"), exist_ok=True)

        self.session.state["run_id"] = run_id
        self.session.state["run_dir"] = run_dir
        return run_dir

    def get_parameters(self):
        """Return the automatically inferred parameters for the function using the dcstring of the function."""
        function_inferrer = FunctionInferrer.infer_from_function_reference(self.run)
        function_json = function_inferrer.to_json_schema()
        parameters = function_json.get("parameters")
        if not parameters:
            raise Exception(
                "Failed to infere parameters, please define JSON instead of using this automated util."
            )

        parameters["properties"].pop("args", None)
        parameters["properties"].pop("kwargs", None)

        if "required" in parameters:
            parameters["required"] = [
                param
                for param in parameters["required"]
                if param not in ["args", "kwargs"]
            ]

        return parameters

    def to_llm_format(self):
        """Convert the agent to LLM tool format."""
        return {
            "name": self.agent_name,
            "description": self.description,
            "parameters": self.parameters,
        }

    @property
    def name(self):
        return self.agent_name

    @property
    def agent_description(self):
        return self.description

    def safe_call(self, *args, **kwargs):
        try:
            return self.run(*args, **kwargs)

        except Exception as e:
            logger.exception(f"error in {self.agent_name} agent: {e}")
            return AgentResponse(status=AgentStatus.ERROR, message=str(e))

    @abstractmethod
    def run(*args, **kwargs) -> AgentResponse:
        pass
