# Copyright (C) 2026 raon-npu authors
# SPDX-License-Identifier: Apache-2.0
# Second prompt for the laptop prefill checks: an English code-review request (Python source), so the MoE routing
# differs from the Korean agent prompt in agent_prompt.py.
_code = '''import json, time
from pathlib import Path


class JobQueue:
    def __init__(self, path):
        self.path = Path(path)
        self.items = json.loads(self.path.read_text()) if self.path.exists() else []

    def push(self, job, priority=0):
        self.items.append({"job": job, "priority": priority, "ts": time.time()})
        self.items.sort(key=lambda item: item["priority"])
        self.path.write_text(json.dumps(self.items))

    def pop(self):
        if not self.items:
            return None
        item = self.items.pop()
        self.path.write_text(json.dumps(self.items))
        return item["job"]

    def retry_failed(self, failed, limit=3):
        for job in failed:
            attempts = job.get("attempts", 0)
            if attempts < limit:
                job["attempts"] = attempts + 1
                self.push(job, priority=-attempts)
'''
messages = [
    {"role": "system", "content": "You are a senior Python reviewer. Point out concrete bugs first, then risky "
                                  "design choices, and keep every finding tied to a line of the code."},
    {"role": "user", "content": "Please review this small persistent job queue. It is used by two worker processes "
                                "on the same host, and jobs sometimes run twice or disappear after a crash.\n\n"
                                "```python\n" + _code + "```\n\nWhat is wrong, and what is the smallest fix for each "
                                "problem? Answer in English."},
]
