"""OpenAI-compatible HTTP server.

Run:  ./sync.sh run 'python -m server.api --port 8000'
Then: curl localhost:8000/v1/completions -d '{"prompt":"Hi","max_tokens":20}'

## Shape of it

The engine step is blocking GPU work, so it runs on its own thread and the
FastAPI handlers never touch the model. A request drops a Sequence into the
scheduler and then waits on its own asyncio.Queue; the engine thread pushes
tokens into that queue as they are produced. That is the whole coupling.

Keeping them apart is what makes continuous batching visible from outside: ten
concurrent requests become one batch inside a single step, and each connection
streams its own tokens as they land.

vLLM and SGLang split these across *processes* (ZeroMQ between them) so
tokenisation and detokenisation overlap the GPU loop on other cores. A thread
is the same idea one notch simpler, and enough while detokenisation is cheap.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.cuda_graph import DecodeGraphRunner
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from engine.sequence import SequenceStatus
from models.qwen3 import Qwen3ForCausalLM


# ---------------------------------------------------------------- streaming

@dataclass
class Stream:
    """One request's output channel, plus incremental detokenisation state."""

    queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    prompt_len: int = 0
    emitted_chars: int = 0

    def delta(self, tokenizer, token_ids: list[int]) -> str:
        """Text produced since the last call.

        Decoding one token at a time is wrong: a multi-byte character spans
        several tokens and would come out as replacement characters. Decode the
        whole generated run each time and return only the new tail.
        """
        text = tokenizer.decode(token_ids[self.prompt_len:], skip_special_tokens=True)
        new = text[self.emitted_chars:]
        self.emitted_chars = len(text)
        return new


class AsyncEngine:
    """LLMEngine on a background thread, with per-request queues."""

    def __init__(self, model, tokenizer, manager, graph_runner=None):
        self.tokenizer = tokenizer
        self.manager = manager
        self.scheduler = Scheduler(manager, max_num_seqs=32,
                                   max_num_batched_tokens=8192)
        self.engine = LLMEngine(model, manager, self.scheduler, graph_runner)

        self.loop: asyncio.AbstractEventLoop | None = None
        self.streams: dict[int, Stream] = {}
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.stop = False
        self.fatal: str | None = None
        self.thread: threading.Thread | None = None

        self.total_requests = 0
        self.total_tokens = 0
        self.started = time.time()

    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def shutdown(self) -> None:
        self.stop = True
        self.wake.set()
        if self.thread:
            self.thread.join(timeout=5)

    def submit(self, prompt_ids: list[int], max_tokens: int,
               temperature: float, top_p: float, top_k: int) -> Stream:
        stream = Stream(prompt_len=len(prompt_ids))
        with self.lock:
            seq = self.engine.add_request(
                prompt_ids, max_tokens, self.tokenizer.eos_token_id,
                temperature=temperature, top_p=top_p, top_k=top_k,
            )
            self.streams[seq.seq_id] = stream
            self.total_requests += 1
        self.wake.set()
        return stream

    def _push(self, stream: Stream, item) -> None:
        """Hand an item to the event loop from the engine thread."""
        assert self.loop is not None
        self.loop.call_soon_threadsafe(stream.queue.put_nowait, item)

    def _run(self) -> None:
        """The engine loop, wrapped so a crash surfaces instead of hanging.

        Without this, an exception kills the daemon thread, no tokens are ever
        pushed, and every in-flight request just times out with no clue why.
        """
        try:
            self._loop_forever()
        except BaseException as exc:  # noqa: BLE001 - report anything
            import traceback
            traceback.print_exc()
            self.fatal = f"{type(exc).__name__}: {exc}"
            with self.lock:
                streams = list(self.streams.values())
                self.streams.clear()
            for stream in streams:
                self._push(stream, {"done": "error", "generated": 0})

    def _loop_forever(self) -> None:
        """One step advances every scheduled sequence."""
        while not self.stop:
            with self.lock:
                has_work = self.scheduler.has_work
                seqs = self.engine.step() if has_work else []

            for seq in seqs:
                stream = self.streams.get(seq.seq_id)
                if stream is None:
                    continue
                text = stream.delta(self.tokenizer, seq.token_ids)
                if text:
                    self._push(stream, text)
                if seq.status is SequenceStatus.FINISHED:
                    self._push(stream, {"done": seq.finish_reason or "stop",
                                        "generated": len(seq.token_ids) - stream.prompt_len})
                    with self.lock:
                        self.streams.pop(seq.seq_id, None)
                        self.total_tokens += len(seq.token_ids) - stream.prompt_len

            if not has_work:
                # Nothing queued: sleep until a request arrives.
                self.wake.clear()
                self.wake.wait(timeout=0.05)


# ---------------------------------------------------------------- API models

class CompletionRequest(BaseModel):
    model: str = "qwen3"
    prompt: str
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "qwen3"
    messages: list[ChatMessage]
    max_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    stream: bool = False


# ---------------------------------------------------------------- app

def build_app(engine_holder: dict) -> FastAPI:
    app = FastAPI(title="mini-llm-inference-server")

    def eng() -> AsyncEngine:
        e = engine_holder.get("engine")
        if e is None:
            raise HTTPException(503, "engine not ready")
        return e

    async def collect(stream: Stream) -> tuple[str, str, int]:
        """Drain a stream to completion. Returns (text, finish_reason, count)."""
        parts = []
        while True:
            item = await stream.queue.get()
            if isinstance(item, dict):
                return "".join(parts), item["done"], item["generated"]
            parts.append(item)

    async def sse(stream: Stream, request_id: str, model: str,
                  chat: bool) -> "asyncio.AsyncIterator[str]":
        created = int(time.time())
        kind = "chat.completion.chunk" if chat else "text_completion"

        def chunk(delta: str | None, finish: str | None) -> str:
            if chat:
                choice = {"index": 0, "finish_reason": finish,
                          "delta": {"content": delta} if delta is not None else {}}
            else:
                choice = {"index": 0, "finish_reason": finish, "text": delta or ""}
            body = {"id": request_id, "object": kind, "created": created,
                    "model": model, "choices": [choice]}
            return f"data: {json.dumps(body)}\n\n"

        if chat:
            yield chunk(None, None)  # OpenAI opens with an empty delta
        while True:
            item = await stream.queue.get()
            if isinstance(item, dict):
                yield chunk(None, item["done"])
                yield "data: [DONE]\n\n"
                return
            yield chunk(item, None)

    @app.get("/health")
    async def health():
        e = engine_holder.get("engine")
        if e is not None and e.fatal:
            raise HTTPException(500, f"engine thread died: {e.fatal}")
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models():
        return {"object": "list",
                "data": [{"id": "qwen3", "object": "model", "owned_by": "local"}]}

    @app.get("/stats")
    async def stats():
        e = eng()
        m = e.manager
        return {
            "uptime_s": round(time.time() - e.started, 1),
            "requests": e.total_requests,
            "generated_tokens": e.total_tokens,
            "in_flight": len(e.streams),
            "kv_blocks_total": m.num_blocks,
            "kv_blocks_free": m.num_free_blocks,
            "kv_kb_per_token": round(m.bytes_per_token() / 1024, 1),
            "prefix_cache_hit_rate": round(m.hit_rate, 4),
        }

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        e = eng()
        prompt_ids = e.tokenizer(req.prompt).input_ids
        stream = e.submit(prompt_ids, req.max_tokens, req.temperature,
                          req.top_p, req.top_k)
        rid = f"cmpl-{uuid.uuid4().hex[:24]}"

        if req.stream:
            return StreamingResponse(sse(stream, rid, req.model, chat=False),
                                     media_type="text/event-stream")
        text, finish, count = await collect(stream)
        return {
            "id": rid, "object": "text_completion", "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "text": text, "finish_reason": finish}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": count,
                      "total_tokens": len(prompt_ids) + count},
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatRequest):
        e = eng()
        # tokenize=False then tokenize separately. With tokenize=True,
        # transformers 5.x hands back a BatchEncoding, and treating that as a
        # list of ids silently yields its string KEYS.
        text = e.tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            add_generation_prompt=True, tokenize=False,
        )
        prompt_ids = e.tokenizer(text, add_special_tokens=False).input_ids
        stream = e.submit(prompt_ids, req.max_tokens, req.temperature,
                          req.top_p, req.top_k)
        rid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

        if req.stream:
            return StreamingResponse(sse(stream, rid, req.model, chat=True),
                                     media_type="text/event-stream")
        text, finish, count = await collect(stream)
        return {
            "id": rid, "object": "chat.completion", "created": int(time.time()),
            "model": req.model,
            "choices": [{"index": 0, "finish_reason": finish,
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": len(prompt_ids), "completion_tokens": count,
                      "total_tokens": len(prompt_ids) + count},
        }

    return app


def load(model_path: str, num_blocks: int, block_size: int, use_graphs: bool):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    hf = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    model.load_state_dict(hf.state_dict())
    model = model.cuda().eval()
    del hf
    torch.cuda.empty_cache()

    manager = build_manager_for(model, num_blocks=num_blocks, block_size=block_size)
    runner = None
    if use_graphs:
        runner = DecodeGraphRunner(model, manager, max_batch=32,
                                   max_blocks_per_seq=manager.num_blocks)
        runner.capture()
    return model, tokenizer, manager, runner


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--num-blocks", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--no-cuda-graphs", action="store_true")
    args = ap.parse_args()

    holder: dict = {}
    app = build_app(holder)

    @app.on_event("startup")
    async def startup():
        print(f"[server] loading {args.model} ...", flush=True)
        model, tokenizer, manager, runner = load(
            args.model, args.num_blocks, args.block_size, not args.no_cuda_graphs)
        engine = AsyncEngine(model, tokenizer, manager, runner)
        engine.start()
        holder["engine"] = engine
        print(f"[server] ready on http://{args.host}:{args.port}  "
              f"kv {manager.capacity_tokens()} tokens, "
              f"cuda_graphs={not args.no_cuda_graphs}", flush=True)

    @app.on_event("shutdown")
    async def shutdown():
        if "engine" in holder:
            holder["engine"].shutdown()

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
