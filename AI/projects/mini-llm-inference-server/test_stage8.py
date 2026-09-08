"""Stage 8 gate: the HTTP server works, streams, and batches concurrent load.

Run: ./sync.sh run 'python test_stage8.py'

Starts the server as a subprocess and drives it over HTTP:

  1. /health and /v1/models answer
  2. /v1/completions returns OpenAI-shaped JSON with usage counts
  3. stream=true yields SSE chunks ending in [DONE], and the concatenated
     deltas equal the non-streamed text for the same greedy request
  4. /v1/chat/completions applies the chat template
  5. 8 concurrent requests are served in fewer engine steps than 8 sequential
     ones would take — the point of continuous batching
  6. /stats reports KV usage
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import time

import httpx

BASE = "http://127.0.0.1:8123"
ARGS = [sys.executable, "-m", "server.api", "--port", "8123",
        "--num-blocks", "64", "--block-size", "256"]


LOG = "/tmp/stage8_server.log"


def start_server():
    print("[test] starting server ...", flush=True)
    # Log to a file, not a pipe. A pipe nobody reads hides the traceback when
    # the engine thread dies, and the symptom is just an unexplained timeout.
    log = open(LOG, "w")
    proc = subprocess.Popen(ARGS, stdout=log, stderr=subprocess.STDOUT,
                            text=True, preexec_fn=os.setsid)
    # Wait for readiness rather than sleeping a fixed time.
    deadline = time.time() + 600
    while time.time() < deadline:
        if proc.poll() is not None:
            print(open(LOG).read())
            raise SystemExit("server died during startup")
        try:
            if httpx.get(f"{BASE}/health", timeout=2).status_code == 200:
                print("[test] server up", flush=True)
                return proc
        except Exception:
            time.sleep(2)
    raise SystemExit("server did not become ready")


async def run_checks():
    async with httpx.AsyncClient(timeout=120) as c:
        ok = True

        # --- 1. basics ---
        assert (await c.get(f"{BASE}/health")).json()["status"] == "ok"
        models = (await c.get(f"{BASE}/v1/models")).json()
        assert models["data"][0]["id"] == "qwen3"
        print("[1] health and /v1/models          ok")

        # --- 2. non-streaming completion ---
        body = {"prompt": "The specialty of Hanoi is", "max_tokens": 24,
                "temperature": 0.0}
        r = (await c.post(f"{BASE}/v1/completions", json=body)).json()
        plain = r["choices"][0]["text"]
        assert plain.strip(), "empty completion"
        assert r["usage"]["completion_tokens"] > 0
        assert r["choices"][0]["finish_reason"] in ("stop", "length")
        print(f"[2] completion ({r['usage']['completion_tokens']} tokens)      ok")
        print(f"    {plain.strip()[:70]!r}")

        # --- 3. streaming must reassemble to the same greedy text ---
        chunks, saw_done = [], False
        async with c.stream("POST", f"{BASE}/v1/completions",
                            json={**body, "stream": True}) as resp:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    saw_done = True
                    break
                chunks.append(json.loads(payload)["choices"][0]["text"])
        streamed = "".join(chunks)
        assert saw_done, "stream never sent [DONE]"
        same = streamed == plain
        ok &= same
        print(f"[3] streaming, {len(chunks)} chunks, [DONE] seen, "
              f"matches non-streamed: {'yes' if same else 'NO'}"
              f"{'' if same else '  FAIL'}")
        if not same:
            print(f"    non-stream: {plain!r}\n    streamed:   {streamed!r}")

        # --- 4. chat template ---
        chat = {"messages": [{"role": "user", "content": "Name one city in Vietnam."}],
                "max_tokens": 24, "temperature": 0.0}
        r = (await c.post(f"{BASE}/v1/chat/completions", json=chat)).json()
        reply = r["choices"][0]["message"]["content"]
        assert reply.strip(), "empty chat reply"
        print(f"[4] chat completion                ok")
        print(f"    {reply.strip()[:70]!r}")

        # --- 5. concurrency: 8 at once should not cost 8x one ---
        one_body = {"prompt": "Count to ten:", "max_tokens": 32, "temperature": 0.0}
        t0 = time.perf_counter()
        await c.post(f"{BASE}/v1/completions", json=one_body)
        solo = time.perf_counter() - t0

        t0 = time.perf_counter()
        await asyncio.gather(*[
            c.post(f"{BASE}/v1/completions",
                   json={**one_body, "prompt": f"Count to ten ({i}):"})
            for i in range(8)
        ])
        eight = time.perf_counter() - t0
        speedup = 8 * solo / eight
        batched = speedup > 2.0
        ok &= batched
        print(f"[5] 1 request {solo:.2f}s, 8 concurrent {eight:.2f}s "
              f"-> {speedup:.1f}x throughput vs sequential"
              f"{'' if batched else '  FAIL (not batching)'}")

        # --- 6. stats ---
        s = (await c.get(f"{BASE}/stats")).json()
        assert s["kv_blocks_free"] == s["kv_blocks_total"], \
            f"blocks leaked: {s['kv_blocks_free']}/{s['kv_blocks_total']}"
        print(f"[6] stats: {s['requests']} requests, {s['generated_tokens']} tokens, "
              f"{s['kv_blocks_free']}/{s['kv_blocks_total']} blocks free   ok")
        return ok


def main():
    proc = start_server()
    try:
        ok = asyncio.run(run_checks())
    except BaseException:
        print("\n--- server log (tail) ---")
        print("".join(open(LOG).readlines()[-40:]))
        raise
    finally:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass

    print()
    if ok:
        print("[stage8] PASS — OpenAI-compatible server with streaming and batching")
        return
    print("[stage8] FAIL")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
