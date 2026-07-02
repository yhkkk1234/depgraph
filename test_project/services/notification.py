from models.task import Task, TaskStatus
from models.user import User
from typing import List


class NotificationService:
    def __init__(self):
        self.notifications = []

    def notify_status_change(self, task: Task, user: User, old_status: TaskStatus):
        message = (
            f"Task '{task.to_dict()['title']}' status changed "
            f"from {old_status.value} to {task.status.value} "
            f"for user {user.username}"
        )
        self.notifications.append(message)
        return message

    def notify_task_assigned(self, task: Task, user: User):
        message = (
            f"Task '{task.to_dict()['title']}' assigned to {user.username}"
        )
        self.notifications.append(message)
        return message

    def get_recent_notifications(self, limit: int = 10) -> List[str]:
        return self.notifications[-limit:]
