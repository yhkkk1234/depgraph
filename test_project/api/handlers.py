from services.task_service import TaskService
from services.notification import NotificationService
from utils.validators import validate_task_data, format_task_for_display
from models.task import TaskStatus, TaskPriority
from models.user import User
from storage.database import Database


class TaskHandlers:
    def __init__(self):
        self.task_service = TaskService()
        self.notif_service = NotificationService()
        self.db = Database.get_instance()

    def handle_create_task(self, data: dict) -> dict:
        errors = validate_task_data(data)
        if errors:
            return {"success": False, "errors": errors}

        task = self.task_service.create_task(
            title=data["title"],
            description=data.get("description", ""),
            priority=TaskPriority(data.get("priority", 2)),
            assignee_id=data.get("assignee_id"),
        )
        return {"success": True, "task": task.to_dict()}

    def handle_get_task(self, task_id: int) -> dict:
        task_data = self.task_service.get_task(task_id)
        if task_data is None:
            return {"success": False, "error": "Task not found"}
        return {"success": True, "task": task_data}

    def handle_list_tasks(self, status: str = None) -> dict:
        task_status = TaskStatus(status) if status else None
        tasks = self.task_service.list_tasks(status=task_status)
        return {"success": True, "tasks": tasks, "count": len(tasks)}

    def handle_update_status(self, task_id: int, new_status: str) -> dict:
        task = self.db.get_task(task_id)
        if task is None:
            return {"success": False, "error": "Task not found"}

        old_status = task.status
        updated = self.task_service.update_status(
            task_id, TaskStatus(new_status)
        )
        if updated is None:
            return {"success": False, "error": "Update failed"}

        assignee = self.db.get_user(task.assignee_id) if task.assignee_id else None
        if assignee:
            self.notif_service.notify_status_change(task, assignee, old_status)

        return {"success": True, "task": updated}

    def handle_get_summary(self) -> dict:
        summary = self.task_service.get_task_summary()
        return {"success": True, "summary": summary}
