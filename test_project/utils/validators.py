from models.task import Task, TaskStatus, TaskPriority


def validate_task_data(data: dict) -> list:
    errors = []
    if not data.get("title"):
        errors.append("title is required")
    elif len(data["title"]) > 200:
        errors.append("title must be <= 200 characters")

    if "status" in data:
        valid_statuses = [s.value for s in TaskStatus]
        if data["status"] not in valid_statuses:
            errors.append(f"status must be one of {valid_statuses}")

    if "priority" in data:
        valid_priorities = [p.value for p in TaskPriority]
        if data["priority"] not in valid_priorities:
            errors.append(f"priority must be one of {valid_priorities}")
    return errors


def format_task_for_display(task: Task) -> str:
    d = task.to_dict()
    return (
        f"[{d['task_id']}] {d['title']} "
        f"({d['status']}, priority={d['priority']})"
    )
