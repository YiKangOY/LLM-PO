from dataclasses import dataclass, asdict
import datetime
from typing import Dict, List, Any, TypedDict, Annotated
import os, logging, json
import threading
from configs import get_agent_logs_dir
@dataclass
class LLMResponse:
    """Structure for logging LLM responses"""
    agent_type: str
    timestamp: datetime
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    processing_time: float
    iteration: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class LLMInteraction:
    """Structure for logging individual LLM message/response pairs"""
    agent_type: str
    timestamp: datetime
    iteration: int
    messages_sent: List[Dict[str, Any]]
    response_received: Dict[str, Any]
    processing_time: float
    model_name: str
    token_usage: Dict[str, Any] = None
    round_index: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class ECOLogger:
    """Centralized logging system for ECO agent responses and LLM interactions"""
    
    def __init__(self, log_file: str = "eco_agent_responses.json", llm_log_file: str = None):
        self.log_file = log_file
        self.llm_log_file = llm_log_file
        self.responses = []
        self.llm_interactions = []
        self._lock = threading.Lock()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = get_agent_logs_dir()
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, os.path.basename(self.log_file))
        if self.llm_log_file is None:
            self.llm_log_file = os.path.join(log_dir, "llm_interactions.json")
        else:
            self.llm_log_file = os.path.join(log_dir, os.path.basename(self.llm_log_file))
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'eco_system.log')),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('ECOSystem')
    
    def log_response(self, response: LLMResponse):
        """Log an LLM response"""
        with self._lock:
            self.responses.append(response)
            self.logger.info(f"Agent {response.agent_type} - Iteration {response.iteration} - Time: {response.processing_time:.2f}s")
            self._save_to_json()
    
    def log_llm_interaction(self, interaction: LLMInteraction):
        """Log an LLM interaction (messages sent and response received)"""
        with self._lock:
            self.llm_interactions.append(interaction)
            self.logger.info(
                f"LLM Interaction - {interaction.agent_type} - Round {interaction.round_index} - "
                f"Iteration {interaction.iteration} - Time: {interaction.processing_time:.2f}s - "
                f"Model: {interaction.model_name}"
            )
            self._save_llm_interactions()
    
    def _save_to_json(self):
        """Save all responses to JSON file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump([resp.to_dict() for resp in self.responses], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save responses to JSON: {e}")
    
    def _save_llm_interactions(self):
        """Save all LLM interactions to JSON file"""
        try:
            with open(self.llm_log_file, 'w') as f:
                json.dump([interaction.to_dict() for interaction in self.llm_interactions], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save LLM interactions to JSON: {e}")
