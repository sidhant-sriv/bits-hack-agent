from typing_extensions import TypedDict
from typing import List, Dict, Any


class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    context: int
    user_data: Dict[str, Any]
    form_struct: Dict[str, Any]
