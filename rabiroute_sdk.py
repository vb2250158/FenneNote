from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


DEFAULT_MANAGER_URL = "http://127.0.0.1:8790"


@dataclass(frozen=True)
class RabiRouteMatch:
    route: dict[str, Any]
    reason: str


class RabiRouteSdk:
    """Small Python client for RabiRoute manager RabiAPI."""

    def __init__(self, manager_url: str = DEFAULT_MANAGER_URL, timeout_seconds: float = 5.0) -> None:
        self.manager_url = self.normalize_manager_url(manager_url)
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def normalize_manager_url(value: str) -> str:
        text = (value or DEFAULT_MANAGER_URL).strip().rstrip("/")
        return text or DEFAULT_MANAGER_URL

    @staticmethod
    def manager_url_from_webhook(webhook_url: str) -> str:
        parsed = urllib.parse.urlparse(webhook_url.strip())
        scheme = parsed.scheme or "http"
        host = parsed.hostname or "127.0.0.1"
        return f"{scheme}://{host}:8790"

    def get_identity(self) -> dict[str, Any]:
        return self._get_data("/api/rabi/identity")

    def get_routes(self, guid: str | None = None) -> list[dict[str, Any]]:
        route_guid = guid or str(self.get_identity().get("guid", ""))
        data = self._get_data(f"/api/rabi/instances/{self._quote(route_guid)}/routes")
        routes = data.get("routes", [])
        return routes if isinstance(routes, list) else []

    def get_agent_options(self, route_id: str, guid: str | None = None) -> dict[str, Any]:
        route_guid = guid or str(self.get_identity().get("guid", ""))
        return self._get_data(
            f"/api/rabi/instances/{self._quote(route_guid)}/routes/{self._quote(route_id)}/agent-options"
        )

    def set_agent_binding(
        self,
        route_id: str,
        *,
        agent_adapter: str = "codex",
        codex_cwd: str | None = None,
        codex_thread_name: str | None = None,
        guid: str | None = None,
    ) -> dict[str, Any]:
        route_guid = guid or str(self.get_identity().get("guid", ""))
        payload: dict[str, Any] = {"agentAdapter": agent_adapter}
        if codex_cwd is not None:
            payload["codexCwd"] = codex_cwd
        if codex_thread_name is not None:
            payload["codexThreadName"] = codex_thread_name
        return self._request_data(
            f"/api/rabi/instances/{self._quote(route_guid)}/routes/{self._quote(route_id)}/agent-binding",
            method="PATCH",
            payload=payload,
        )

    def match_route_for_webhook(self, webhook_url: str, routes: list[dict[str, Any]]) -> RabiRouteMatch | None:
        parsed = urllib.parse.urlparse(webhook_url.strip())
        if not parsed.port:
            return None
        path = parsed.path or "/"
        exact_matches: list[dict[str, Any]] = []
        port_matches: list[dict[str, Any]] = []
        for route in routes:
            for candidate_port, candidate_path in self.route_webhook_candidates(route):
                if candidate_port != parsed.port:
                    continue
                port_matches.append(route)
                if self._normalize_path(candidate_path) == self._normalize_path(path):
                    exact_matches.append(route)
        if exact_matches:
            return RabiRouteMatch(exact_matches[0], "端口和路径匹配")
        if port_matches:
            return RabiRouteMatch(port_matches[0], "端口匹配")
        return None

    @classmethod
    def route_display_name(cls, route: dict[str, Any]) -> str:
        for key in ("name", "routeName", "configName", "id"):
            value = route.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "未命名路由"

    @classmethod
    def route_id(cls, route: dict[str, Any]) -> str:
        value = route.get("id")
        return value if isinstance(value, str) else ""

    @classmethod
    def route_webhook_candidates(cls, route: dict[str, Any]) -> list[tuple[int, str]]:
        runtime = route.get("runtimeStatus")
        if not isinstance(runtime, dict):
            runtime = {}
        candidates: list[tuple[int, str]] = []
        for port_key, path_key, default_path in (
            ("fenneNoteWebhookPort", "fenneNoteWebhookPath", "/fennenote"),
            ("webhookPort", "webhookPath", "/webhook"),
            ("gatewayPort", "webhookPath", "/webhook"),
        ):
            port = cls._as_int(route.get(port_key, runtime.get(port_key)))
            if port is None:
                continue
            path_value = route.get(path_key, runtime.get(path_key, default_path))
            path = path_value if isinstance(path_value, str) and path_value.strip() else default_path
            candidates.append((port, path))
        return candidates

    def _get_data(self, path: str) -> dict[str, Any]:
        return self._request_data(path, method="GET", payload=None)

    def _request_data(self, path: str, *, method: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        url = f"{self.manager_url}{path}"
        body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Accept": "application/json",
                "User-Agent": "FenneNote-RabiRoute-SDK",
            },
        )
        if body is not None:
            request.add_header("Content-Type", "application/json; charset=utf-8")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                text = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            message = self._extract_error_message(text) or f"HTTP {exc.code}"
            raise RuntimeError(message) from exc
        data = json.loads(text or "{}")
        if not isinstance(data, dict):
            raise RuntimeError("RabiRoute 返回内容不是 JSON 对象。")
        if data.get("code", 0) not in (0, "0", None):
            raise RuntimeError(str(data.get("message") or data.get("error") or "RabiRoute API 返回失败。"))
        result = data.get("data", data)
        if not isinstance(result, dict):
            raise RuntimeError("RabiRoute API data 字段不是 JSON 对象。")
        return result

    @staticmethod
    def _extract_error_message(text: str) -> str:
        try:
            data = json.loads(text or "{}")
        except json.JSONDecodeError:
            return text[:200]
        if isinstance(data, dict):
            return str(data.get("message") or data.get("error") or "")[:200]
        return text[:200]

    @staticmethod
    def _quote(value: str) -> str:
        return urllib.parse.quote(value, safe="")

    @staticmethod
    def _normalize_path(value: str) -> str:
        text = value.strip() or "/"
        return text if text.startswith("/") else f"/{text}"

    @staticmethod
    def _as_int(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
