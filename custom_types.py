from typing import TypedDict
from typing import Literal


class Account(TypedDict):
    id: str
    name: str
    handle: str
    twitter_id: str
    party: Literal['Democrat', 'Independent', 'Republican']