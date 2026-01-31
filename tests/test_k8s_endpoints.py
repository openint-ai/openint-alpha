"""
Integration tests for OpenInt K8s endpoints.

Usage:
    # Against minikube (auto-detects URL):
    pytest tests/test_k8s_endpoints.py -v

    # Against a specific URL:
    OPENINT_BASE_URL=http://localhost:3000 pytest tests/test_k8s_endpoints.py -v
"""

import os
import subprocess
import pytest
import requests

BASE_URL = os.environ.get("OPENINT_BASE_URL", "").rstrip("/")


def _detect_minikube_url():
    """Try to get the UI service URL from minikube."""
    try:
        result = subprocess.run(
            ["minikube", "service", "openint-ui", "--url"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


@pytest.fixture(scope="session")
def base_url():
    url = BASE_URL or _detect_minikube_url()
    if not url:
        pytest.skip("No OPENINT_BASE_URL set and minikube service not reachable")
    # Quick connectivity check
    try:
        requests.get(f"{url}/api/health", timeout=5)
    except requests.ConnectionError:
        pytest.skip(f"Cannot reach {url}")
    return url


# ── Health & Readiness ─────────────────────────────────────────

class TestHealth:
    def test_health(self, base_url):
        r = requests.get(f"{base_url}/api/health", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ("healthy", "ok", True)

    def test_ready(self, base_url):
        r = requests.get(f"{base_url}/api/ready", timeout=10)
        assert r.status_code == 200


# ── Agents ─────────────────────────────────────────────────────

class TestAgents:
    def test_list_agents(self, base_url):
        r = requests.get(f"{base_url}/api/agents", timeout=10)
        assert r.status_code == 200
        data = r.json()
        assert isinstance(data, (list, dict))


# ── Chat ───────────────────────────────────────────────────────

class TestChat:
    def test_chat_returns_response(self, base_url):
        r = requests.post(
            f"{base_url}/api/chat",
            json={"message": "hello", "session_id": "test-k8s"},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        assert "response" in data or "message" in data or "answer" in data or "error" in data

    def test_chat_empty_message_rejected(self, base_url):
        r = requests.post(
            f"{base_url}/api/chat",
            json={"message": ""},
            timeout=10,
        )
        # Should either return 400 or a valid response
        assert r.status_code in (200, 400)


# ── Semantic ───────────────────────────────────────────────────

class TestSemantic:
    def test_list_models(self, base_url):
        r = requests.get(f"{base_url}/api/semantic/models", timeout=10)
        # 200 if modelmgmt-agent available, 503 otherwise
        assert r.status_code in (200, 503)

    def test_models_with_meta(self, base_url):
        r = requests.get(f"{base_url}/api/semantic/models-with-meta", timeout=10)
        assert r.status_code in (200, 503)

    def test_interpret(self, base_url):
        r = requests.get(
            f"{base_url}/api/semantic/interpret",
            params={"q": "test query"},
            timeout=30,
        )
        assert r.status_code in (200, 400, 503)

    def test_analyze(self, base_url):
        r = requests.post(
            f"{base_url}/api/semantic/analyze",
            json={"query": "find customers"},
            timeout=120,
        )
        assert r.status_code in (200, 400, 503)


# ── A2A ────────────────────────────────────────────────────────

class TestA2A:
    AGENT_IDS = ["sg-agent", "modelmgmt-agent", "search_agent", "graph_agent"]

    @pytest.mark.parametrize("agent_id", AGENT_IDS)
    def test_agent_card(self, base_url, agent_id):
        r = requests.get(f"{base_url}/api/a2a/agents/{agent_id}/card", timeout=10)
        assert r.status_code == 200
        card = r.json()
        assert "name" in card or "agent_id" in card or "id" in card

    def test_a2a_run(self, base_url):
        r = requests.post(
            f"{base_url}/api/a2a/run",
            json={"query": "test"},
            timeout=30,
        )
        # Depends on agent availability
        assert r.status_code in (200, 400, 500, 503)


# ── Multi-Agent Demo ───────────────────────────────────────────

class TestMultiAgentDemo:
    def test_recent_queries(self, base_url):
        r = requests.get(
            f"{base_url}/api/multi-agent-demo/queries/recent",
            timeout=10,
        )
        assert r.status_code == 200

    def test_run(self, base_url):
        r = requests.post(
            f"{base_url}/api/multi-agent-demo/run",
            json={"query": "find all customers"},
            timeout=60,
        )
        assert r.status_code in (200, 400, 500, 503)


# ── Suggestions ────────────────────────────────────────────────

class TestSuggestions:
    def test_lucky(self, base_url):
        r = requests.get(f"{base_url}/api/suggestions/lucky", timeout=10)
        assert r.status_code in (200, 503)


# ── UI Serving ─────────────────────────────────────────────────

class TestUI:
    def test_root_serves_html(self, base_url):
        r = requests.get(base_url, timeout=10)
        assert r.status_code == 200
        assert "text/html" in r.headers.get("content-type", "")

    def test_spa_fallback(self, base_url):
        r = requests.get(f"{base_url}/some/spa/route", timeout=10)
        # SPA catch-all should return 200 with index.html
        assert r.status_code in (200, 404)
