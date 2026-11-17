#!/usr/bin/env python3
"""Serve vLLM's Mooncake prefill/decode proxy with the endpoints this harness needs.

tools/vllm-proxy/mooncake_connector_proxy.py is an unmodified copy of vLLM v0.31.0
examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py (db9527a); the vLLM
package does not install it. The harness also needs two endpoints that it lacks:

- GET /health: 200 once the proxy knows the prefill instances and every backend is healthy;
  serve.sbatch waits on it.
- POST /tokenize: forwarded to a decode instance; benchmark.py checks token counts with it.

Arguments are the upstream proxy's: --prefill URL BOOTSTRAP_PORT --decode URL --host --port.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, Response

PROXY = Path(__file__).resolve().parents[1] / "tools" / "vllm-proxy" / "mooncake_connector_proxy.py"
spec = importlib.util.spec_from_file_location("mooncake_connector_proxy", PROXY)
assert spec and spec.loader
proxy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proxy)
app = proxy.app


@app.get("/health")
async def health() -> Response:
    if not app.state.ready.is_set():
        return JSONResponse({"status": "waiting for prefill instances"}, status_code=503)
    for backend in app.state.prefill_clients + app.state.decode_clients:
        try:
            response = await backend["client"].get("/health")
        except Exception as error:  # noqa: BLE001 - any backend failure means not healthy
            return JSONResponse({"status": f"backend unreachable: {error}"}, status_code=503)
        if response.status_code != 200:
            return JSONResponse({"status": "backend unhealthy"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/tokenize")
async def tokenize(request: Request) -> Response:
    client = app.state.decode_clients[0]["client"]
    response = await client.post("/tokenize", content=await request.body(),
                                 headers={"Content-Type": "application/json"})
    return Response(response.content, response.status_code, media_type="application/json")


if __name__ == "__main__":
    # The upstream lifespan reads the module global global_args.
    proxy.global_args = proxy.parse_args()
    uvicorn.run(app, host=proxy.global_args.host, port=proxy.global_args.port)
