from typing import Any, Optional
from dataclasses import dataclass, field



@dataclass
class GraphMemoryItem:
    
    task: str
    agent: str
    expected_output: str = ""
    datetime: Optional[str] = None  # ISO recommended
    metadata: dict[str, Any] = field(default_factory=dict)

    # graph payload
    triplets: list[Any] = field(default_factory=list)   # e.g. [[h,t,{"label":r}], ...]
    episodic: dict[str, Any] = field(default_factory=dict)  # optional

