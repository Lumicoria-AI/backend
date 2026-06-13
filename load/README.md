# Load testing

`locustfile.py` runs a mixed-workload smoke against the Lumicoria API.

## Quick start

```bash
pip install locust
export LUMICORIA_API_URL=http://localhost:8000/api/v1
export LUMICORIA_TEST_EMAIL=loadtest@lumicoria.ai
export LUMICORIA_TEST_PASSWORD='Demo-Password-123!'
export LUMICORIA_ORG_ID=<your org id>
export LUMICORIA_PROJECT_ID=<your project id>

locust -f backend/load/locustfile.py \
  --host=http://localhost:8000 \
  --users 25 --spawn-rate 5 --run-time 2m \
  --headless --only-summary
```

## Workload mix

| Weight | Endpoint |
|---|---|
| 20 | GET `/workspace/:org/timeline` |
| 15 | GET `/notifications/unread/count` |
| 10 | GET `/organizations/:org/projects` |
| 8 | GET `/tasks?project_id=…` |
| 5 | GET `/organizations/:org/projects/:id/activity` |
| 4 | GET `/activity/me/audit` |
| 3 | GET `/permissions/me` |
| 3 | POST `/tasks` |
| 2 | POST `/chat/send` |
| 1 | POST `/agents/document/run` (low — costs LLM tokens) |

## What this catches

Smoke-grade: obvious regressions like N+1 queries, blocking handlers,
auth misses on hot paths.  Not a perf-spec — production capacity work
should layer a real backing infra and a longer (>15 min) soak.

## Pass/fail heuristic for CI

End-of-run JSON includes `p95_ms` and `fail_ratio`.  Suggested gates
for a 2-minute, 25-user smoke:

  fail_ratio < 0.01
  p95_ms < 800
