from typing import TypedDict, List, Optional, Dict, Any
from datetime import datetime

class Message(TypedDict):
    role: str
    content: str

class ConversationContext(TypedDict):
    recent_messages: List[Dict[str, Any]]
    summary: str

class ConversationState(TypedDict):
    user_id: str
    message: str
    context: Optional[List[str]]
    summary: str
    response: Optional[str]
    timestamp: datetime
