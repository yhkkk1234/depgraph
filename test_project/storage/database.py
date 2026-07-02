from models.task import Task
from models.user import User
from typing import Dict, List


class Database:
    _instance = None

    def __init__(self):
        self.tasks: Dict[int, Task] = {}
        self.users: Dict[int, User] = {}
        self._next_task_id = 1
        self._next_user_id = 1

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def add_task(self, task: Task) -> Task:
        task.task_id = self._next_task_id
        self._next_task_id += 1
        self.tasks[task.task_id] = task
        return task

    def get_task(self, task_id: int) -> Task:
        return self.tasks.get(task_id)

    def get_all_tasks(self) -> List[Task]:
        return list(self.tasks.values())

    def update_task(self, task: Task):
        self.tasks[task.task_id] = task

    def delete_task(self, task_id: int):
        self.tasks.pop(task_id, None)

    def add_user(self, user: User) -> User:
        self.users[user.user_id] = user
        return user

    def get_user(self, user_id: int) -> User:
        return self.users.get(user_id)
