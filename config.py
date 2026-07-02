"""
Experiment configuration.
Uses opencode's built-in MiMo v2.5 token-plan API (free, multi-modal).
"""
import os


# ── MiMo v2.5 (token-plan, built into opencode) ──────────────
API_KEY = os.environ.get(
    "EXPERIMENT_API_KEY",
    "your-api-key-here"
)
API_BASE = os.environ.get(
    "EXPERIMENT_API_BASE",
    "https://token-plan-cn.xiaomimimo.com/v1"
)
MODEL_NAME = os.environ.get("EXPERIMENT_MODEL", "mimo-v2.5")

# ── Alternative providers ──────────────────────────────────
# Uncomment and set your own key:
# API_KEY = os.environ.get("EXPERIMENT_API_KEY", "your-key")
# API_BASE = os.environ.get("EXPERIMENT_API_BASE", "https://api.example.com/v1")
# MODEL_NAME = os.environ.get("EXPERIMENT_MODEL", "model-name")

# ── Experiment Settings ───────────────────────────────────────

TEST_PROJECT = os.path.join(os.path.dirname(__file__), "test_project")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# ── Scoring Metrics ───────────────────────────────────────────
EXPECTED_FIXES = {
    "services/task_service.py": [
        "task.to_dict()",
        "t.to_dict()",
        "task.to_dict()",
        "t.to_dict()",
    ],
    "services/notification.py": [
        "task.to_dict()['title']",
        "task.to_dict()['title']",
    ],
    "api/handlers.py": [
        "task.to_dict()",
    ],
    "utils/validators.py": [
        "task.to_dict()",
    ],
}
TOTAL_FIX_SITES = sum(len(v) for v in EXPECTED_FIXES.values())
