from models.task import Task, TaskStatus, TaskPriority
from models.user import User
from storage.database import Database
from typing import List, Optional


class TaskService:
    def __init__(self):
        self.db = Database.get_instance()

    def create_task(self, title: str, description: str = "",
                    priority: TaskPriority = TaskPriority.MEDIUM,
                    assignee_id: Optional[int] = None) -> Task:
        task = Task(
            task_id=0,
            title=title,
            description=description,
            priority=priority,
            assignee_id=assignee_id,
        )
        self.db.add_task(task)
        return task

    def get_task(self, task_id: int) -> Optional[dict]:
        task = self.db.get_task(task_id)
        if task is None:
            return None
        return task.to_dict()

    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[dict]:
        tasks = self.db.get_all_tasks()
        if status:
            tasks = [t for t in tasks if t.status == status]
        return [t.to_dict() for t in tasks]

    def update_status(self, task_id: int, new_status: TaskStatus) -> Optional[dict]:
        task = self.db.get_task(task_id)
        if task is None:
            return None
        task.status = new_status
        self.db.update_task(task)
        return task.to_dict()

    def get_user_tasks(self, user_id: int) -> List[dict]:
        tasks = self.db.get_all_tasks()
        user_tasks = [t for t in tasks if t.assignee_id == user_id]
        return [t.to_dict() for t in user_tasks]

    def get_task_summary(self) -> dict:
        tasks = self.db.get_all_tasks()
        total = len(tasks)
        by_status = {}
        for t in tasks:
            status_name = t.status.value
            by_status[status_name] = by_status.get(status_name, 0) + 1
        return {"total": total, "by_status": by_status}
