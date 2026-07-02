from dataclasses import dataclass
from typing import Optional


@dataclass
class User:
    user_id: int
    username: str
    email: str
    role: str = "member"
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "is_active": self.is_active,
        }
