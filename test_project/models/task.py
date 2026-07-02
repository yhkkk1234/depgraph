from enum import Enum
from datetime import datetime
from typing import Optional


class TaskStatus(Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4


class Task:
    def __init__(self, task_id: int, title: str, description: str = "",
                 status: TaskStatus = TaskStatus.TODO,
                 priority: TaskPriority = TaskPriority.MEDIUM,
                 assignee_id: Optional[int] = None,
                 created_at: Optional[datetime] = None):
        self.task_id = task_id
        self.title = title
        self.description = description
        self.status = status
        self.priority = priority
        self.assignee_id = assignee_id
        self.created_at = created_at or datetime.now()

    def to_dict(self, include_comments: bool) -> dict:
        result = {
            "task_id": self.task_id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "assignee_id": self.assignee_id,
            "created_at": self.created_at.isoformat(),
        }
        if include_comments:
            result["comments"] = []
        return result

    def __repr__(self):
        return f"Task(id={self.task_id}, title='{self.title}', status={self.status.value})"
