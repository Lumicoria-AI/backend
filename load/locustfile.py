"""
Lumicoria AI — Locust smoke load test.

Drives realistic mixed traffic against the API: workspace home reads,
chat sends, agent runs, task creates, notification polls.  Designed to
catch obvious regressions (N+1 queries, blocking handlers, cold-start
auth) rather than to certify a specific RPS.

Usage:

    pip install locust
    LUMICORIA_API_URL=http://localhost:8000/api/v1 \\
    LUMICORIA_TEST_EMAIL=test@example.com \\
    LUMICORIA_TEST_PASSWORD=… \\
    locust -f backend/load/locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to set users + spawn rate.  For a
headless smoke run:

    locust -f backend/load/locustfile.py --host=http://localhost:8000 \\
        --users 25 --spawn-rate 5 --run-time 2m --headless --only-summary
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Optional

from locust import HttpUser, task, between, events


API_BASE = os.getenv("LUMICORIA_API_URL", "/api/v1").rstrip("/")
TEST_EMAIL = os.getenv("LUMICORIA_TEST_EMAIL", "loadtest@lumicoria.ai")
TEST_PASSWORD = os.getenv("LUMICORIA_TEST_PASSWORD", "Demo-Password-123!")
PROJECT_ID = os.getenv("LUMICORIA_PROJECT_ID")
ORG_ID = os.getenv("LUMICORIA_ORG_ID")


class LumicoriaUser(HttpUser):
    """Mixed-workload virtual user.

    Login once at start; thereafter mix of light reads, medium-cost
    writes, and the heavy agent-run path (low weight to avoid hammering
    LLM providers during a smoke run).
    """

    wait_time = between(1.5, 4.0)

    # ── Lifecycle ────────────────────────────────────────────────

    def on_start(self):
        self.access_token: Optional[str] = None
        self.user_id: Optional[str] = None
        self.org_id: Optional[str] = ORG_ID
        self.project_id: Optional[str] = PROJECT_ID
        self._login()

    def _login(self):
        # OAuth2 password grant (form-encoded).  Falls back to a JSON
        # login if the deployment exposes that variant instead.
        with self.client.post(
            f"{API_BASE}/auth/login",
            data={"username": TEST_EMAIL, "password": TEST_PASSWORD},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            catch_response=True,
            name="POST /auth/login",
        ) as resp:
            if resp.status_code != 200:
                # try JSON variant
                resp2 = self.client.post(
                    f"{API_BASE}/auth/login",
                    json={"email": TEST_EMAIL, "password": TEST_PASSWORD},
                    name="POST /auth/login (json)",
                )
                if resp2.status_code != 200:
                    resp.failure(f"login failed: {resp.status_code}")
                    return
                payload = resp2.json()
            else:
                payload = resp.json()
            self.access_token = payload.get("access_token") or payload.get("token")
            user = payload.get("user") or {}
            self.user_id = user.get("id") or user.get("_id") or payload.get("user_id")
            self.org_id = self.org_id or user.get("organization_id")

    @property
    def auth_headers(self) -> dict:
        return {"Authorization": f"Bearer {self.access_token}"} if self.access_token else {}

    # ── Tasks (weighted by realistic frequency) ─────────────────

    @task(20)
    def workspace_home(self):
        if not self.org_id:
            return
        self.client.get(
            f"{API_BASE}/workspace/{self.org_id}/timeline",
            headers=self.auth_headers,
            name="GET /workspace/:org/timeline",
        )

    @task(15)
    def notifications_count(self):
        self.client.get(
            f"{API_BASE}/notifications/unread/count",
            headers=self.auth_headers,
            name="GET /notifications/unread/count",
        )

    @task(10)
    def list_projects(self):
        if not self.org_id:
            return
        self.client.get(
            f"{API_BASE}/organizations/{self.org_id}/projects",
            headers=self.auth_headers,
            name="GET /organizations/:org/projects",
        )

    @task(8)
    def list_tasks(self):
        if not self.project_id:
            return
        self.client.get(
            f"{API_BASE}/tasks?project_id={self.project_id}&limit=25",
            headers=self.auth_headers,
            name="GET /tasks?project_id",
        )

    @task(5)
    def project_activity(self):
        if not self.org_id or not self.project_id:
            return
        self.client.get(
            f"{API_BASE}/organizations/{self.org_id}/projects/{self.project_id}/activity?limit=25",
            headers=self.auth_headers,
            name="GET /organizations/:org/projects/:id/activity",
        )

    @task(4)
    def my_audit(self):
        self.client.get(
            f"{API_BASE}/activity/me/audit?limit=25",
            headers=self.auth_headers,
            name="GET /activity/me/audit",
        )

    @task(3)
    def my_permissions(self):
        params = f"?organization_id={self.org_id}" if self.org_id else ""
        self.client.get(
            f"{API_BASE}/permissions/me{params}",
            headers=self.auth_headers,
            name="GET /permissions/me",
        )

    @task(3)
    def create_task(self):
        if not self.project_id:
            return
        title = f"Load-test task {random.randint(1, 1_000_000)}"
        self.client.post(
            f"{API_BASE}/tasks",
            json={
                "title": title,
                "project_id": self.project_id,
                "priority": random.choice(["low", "medium", "high"]),
                "status": "todo",
            },
            headers={**self.auth_headers, "Content-Type": "application/json"},
            name="POST /tasks",
        )

    @task(2)
    def chat_send(self):
        self.client.post(
            f"{API_BASE}/chat/send",
            json={"message": f"Hello @{random.randint(0, 9999)}"},
            headers={**self.auth_headers, "Content-Type": "application/json"},
            name="POST /chat/send",
        )

    # Heavy / costly — keep weight low for smoke
    @task(1)
    def agent_run(self):
        self.client.post(
            f"{API_BASE}/agents/document/run",
            json={"prompt": "Summarise the latest uploaded document in 3 bullets."},
            headers={**self.auth_headers, "Content-Type": "application/json"},
            name="POST /agents/document/run",
        )


# ── Optional: tag the run with metadata ────────────────────────────


@events.test_start.add_listener
def on_test_start(environment, **_kwargs):
    print("─" * 60)
    print(f"Lumicoria smoke load — API {API_BASE} — user {TEST_EMAIL}")
    print(f"org_id={ORG_ID or 'auto'} project_id={PROJECT_ID or 'unset'}")
    print("─" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **_kwargs):
    stats = environment.stats.total
    print(json.dumps({
        "requests": stats.num_requests,
        "failures": stats.num_failures,
        "fail_ratio": (stats.num_failures / stats.num_requests) if stats.num_requests else 0.0,
        "median_ms": stats.median_response_time,
        "p95_ms": stats.get_response_time_percentile(0.95),
        "p99_ms": stats.get_response_time_percentile(0.99),
        "max_ms": stats.max_response_time,
    }, indent=2))
