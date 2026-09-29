# -*- coding: utf-8 -*-
"""Location: ./tests/live_gateway/mcp/test_identity_propagation_live.py
Copyright contributors to the MCP-CONTEXT-FORGE project
SPDX-License-Identifier: Apache-2.0

Live black-box test for identity propagation to upstream MCP servers (#6855).

The test starts a header-echo MCP server, registers it with a per-gateway
``identity_propagation`` override, calls its tool through ``/servers/{id}/mcp``,
and checks the identity headers the upstream received.

Requirements:
    - A running gateway at ``MCP_CLI_BASE_URL`` (default ``http://127.0.0.1:8080``).
    - The gateway can reach the echo server at ``IDENTITY_ECHO_URL``
      (default ``http://127.0.0.1:<IDENTITY_ECHO_PORT>/mcp``). For a gateway on
      the host, start it with ``SSRF_ALLOW_LOCALHOST=true``. For a compose stack,
      set ``IDENTITY_ECHO_URL`` to an address the container can reach.
    - The Python MCP transport serves ``/mcp`` (the Rust runtime is out of scope).
"""

# Future
from __future__ import annotations

# Standard
import json
import os
import socket
import threading
import time
from typing import Any, Iterator
import uuid

# Third-Party
import httpx
import pytest

# First-Party
from tests.helpers.auth import make_test_jwt

from ..helpers.mcp_test_helpers import ADMIN_EMAIL, BASE_URL, JWT_SECRET, skip_no_gateway

pytestmark = [skip_no_gateway]

ECHO_PORT = int(os.getenv("IDENTITY_ECHO_PORT", "9765"))
ECHO_URL = os.getenv("IDENTITY_ECHO_URL", f"http://127.0.0.1:{ECHO_PORT}/mcp")
ECHO_TOOL = "echo_identity"


def _echo_app():
    """Build a minimal streamable-HTTP MCP server that echoes identity headers.

    Returns:
        A Starlette application answering MCP JSON-RPC with JSON responses.
    """
    # Third-Party
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
    from starlette.routing import Route

    async def mcp_endpoint(request: Request) -> Response:
        message = await request.json()
        method = message.get("method")
        if "id" not in message:
            return Response(status_code=202)
        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": message.get("params", {}).get("protocolVersion", "2025-03-26"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "identity-echo", "version": "1.0.0"},
            }
        elif method == "tools/list":
            result = {"tools": [{"name": ECHO_TOOL, "description": "Echo identity headers", "inputSchema": {"type": "object", "properties": {}}}]}
        elif method == "tools/call":
            received = {name.lower(): value for name, value in request.headers.items() if name.lower().startswith(("x-forwarded-user", "x-sugar-user"))}
            meta_user = (message.get("params", {}).get("_meta") or {}).get("user")
            result = {"content": [{"type": "text", "text": json.dumps({"headers": received, "meta_user": meta_user})}], "isError": False}
        elif method == "ping":
            result = {}
        else:
            return JSONResponse({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": f"unknown method {method}"}})
        return JSONResponse({"jsonrpc": "2.0", "id": message["id"], "result": result})

    return Starlette(routes=[Route("/mcp", mcp_endpoint, methods=["POST"])])


@pytest.fixture(scope="module")
def echo_server() -> Iterator[str]:
    """Run the echo MCP server in a background thread for the module."""
    # Third-Party
    import uvicorn

    server = uvicorn.Server(uvicorn.Config(_echo_app(), host="0.0.0.0", port=ECHO_PORT, log_level="warning"))  # nosec B104 - test fixture must be reachable from a containerised gateway
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", ECHO_PORT)) == 0:
                break
        time.sleep(0.1)
    else:
        pytest.fail(f"echo server did not start on port {ECHO_PORT}")
    yield ECHO_URL
    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture(scope="module")
def admin_headers() -> dict[str, str]:
    """Admin bearer token headers for REST setup calls."""
    token = make_test_jwt(ADMIN_EMAIL, is_admin=True, teams=None, secret=JWT_SECRET)
    return {"Authorization": f"Bearer {token}"}


def _register(admin_headers: dict[str, str], echo_url: str, identity_propagation: dict[str, Any], passthrough: list[str] | None = None) -> tuple[str, str]:
    """Register the echo gateway and a virtual server exposing its tool.

    Args:
        admin_headers: Admin authorization headers.
        echo_url: Upstream MCP URL.
        identity_propagation: Per-gateway identity propagation override.
        passthrough: Optional per-gateway passthrough header allow-list.

    Returns:
        Tuple of (gateway_id, server_id).
    """
    suffix = uuid.uuid4().hex[:8]
    with httpx.Client(base_url=BASE_URL, headers=admin_headers, timeout=30) as client:
        body: dict[str, Any] = {
            "name": f"identity-echo-{suffix}",
            "url": echo_url,
            "transport": "STREAMABLEHTTP",
            "visibility": "public",
            "identity_propagation": identity_propagation,
        }
        if passthrough:
            body["passthrough_headers"] = passthrough
        response = client.post("/gateways", json=body)
        assert response.status_code in (200, 201), response.text
        gateway_id = response.json()["id"]

        tool_ids: list[str] = []
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline and not tool_ids:
            tools = client.get("/tools", params={"gateway_id": gateway_id, "include_pagination": "false"}).json()
            tools = tools.get("items", tools) if isinstance(tools, dict) else tools
            tool_ids = [tool["id"] for tool in tools if tool.get("gatewayId", tool.get("gateway_id")) == gateway_id]
            if not tool_ids:
                time.sleep(1)
        assert tool_ids, "echo tool was not discovered"

        response = client.post("/servers", json={"server": {"name": f"identity-echo-vs-{suffix}", "associated_tools": tool_ids, "visibility": "public"}})
        assert response.status_code in (200, 201), response.text
        return gateway_id, response.json()["id"]


def _cleanup(admin_headers: dict[str, str], gateway_id: str, server_id: str) -> None:
    with httpx.Client(base_url=BASE_URL, headers=admin_headers, timeout=30) as client:
        client.delete(f"/servers/{server_id}")
        client.delete(f"/gateways/{gateway_id}")


async def _call_echo(server_id: str, token: str, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    """Call the echo tool through the virtual server's MCP endpoint.

    Args:
        server_id: Virtual server ID.
        token: Caller bearer token.
        extra_headers: Additional client request headers.

    Returns:
        The JSON payload the echo tool returned.
    """
    # Third-Party
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    headers = {"Authorization": f"Bearer {token}", **(extra_headers or {})}
    async with streamable_http_client(f"{BASE_URL}/servers/{server_id}/mcp/", http_client=create_mcp_http_client(headers=headers, timeout=httpx2.Timeout(15.0))) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            name = next(tool.name for tool in tools.tools if tool.name.endswith(ECHO_TOOL.replace("_", "-")) or tool.name.endswith(ECHO_TOOL))
            result = await session.call_tool(name, {})
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_authenticated_identity_reaches_upstream(echo_server: str, admin_headers: dict[str, str]) -> None:
    """The upstream receives the caller's identity from ContextForge."""
    gateway_id, server_id = _register(admin_headers, echo_server, {"enabled": True, "mode": "both"})
    try:
        payload = await _call_echo(server_id, admin_headers["Authorization"].split(" ", 1)[1])
    finally:
        _cleanup(admin_headers, gateway_id, server_id)

    assert payload["headers"]["x-forwarded-user-email"] == ADMIN_EMAIL
    assert payload["headers"]["x-forwarded-user-id"] == ADMIN_EMAIL
    assert payload["meta_user"]["email"] == ADMIN_EMAIL


@pytest.mark.asyncio
async def test_gateway_opt_out_sends_no_identity(echo_server: str, admin_headers: dict[str, str]) -> None:
    """A gateway with identity propagation disabled receives no identity."""
    gateway_id, server_id = _register(admin_headers, echo_server, {"enabled": False})
    try:
        payload = await _call_echo(server_id, admin_headers["Authorization"].split(" ", 1)[1])
    finally:
        _cleanup(admin_headers, gateway_id, server_id)

    assert payload["headers"] == {}
    assert payload["meta_user"] is None


@pytest.mark.asyncio
async def test_client_cannot_spoof_identity_header(echo_server: str, admin_headers: dict[str, str]) -> None:
    """A client-sent identity header never replaces the authenticated identity."""
    gateway_id, server_id = _register(admin_headers, echo_server, {"enabled": True, "mode": "headers"}, passthrough=["X-Forwarded-User-Email"])
    try:
        payload = await _call_echo(server_id, admin_headers["Authorization"].split(" ", 1)[1], {"X-Forwarded-User-Email": "ceo@example.com"})
    finally:
        _cleanup(admin_headers, gateway_id, server_id)

    assert payload["headers"]["x-forwarded-user-email"] == ADMIN_EMAIL
