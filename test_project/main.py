from models.user import User
from storage.database import Database
from api.handlers import TaskHandlers


def seed_data():
    db = Database.get_instance()
    db.add_user(User(user_id=1, username="alice", email="alice@example.com"))
    db.add_user(User(user_id=2, username="bob", email="bob@example.com"))


def main():
    seed_data()
    handlers = TaskHandlers()

    print("=== Create Tasks ===")
    r1 = handlers.handle_create_task({"title": "Fix login bug", "priority": 3, "assignee_id": 1})
    print(r1)
    r2 = handlers.handle_create_task({"title": "Add dark mode", "priority": 2, "assignee_id": 2})
    print(r2)

    print("\n=== List Tasks ===")
    r3 = handlers.handle_list_tasks()
    print(r3)

    print("\n=== Update Status ===")
    r4 = handlers.handle_update_status(1, "in_progress")
    print(r4)

    print("\n=== Summary ===")
    r5 = handlers.handle_get_summary()
    print(r5)


if __name__ == "__main__":
    main()
