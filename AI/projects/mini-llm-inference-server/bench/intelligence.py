"""Does the server produce *correct* output, not just fast output?

Run: ./sync.sh run 'python -m bench.intelligence --task gsm8k --n 200'
     ./sync.sh run 'python -m bench.intelligence --task math500 --n 200'

Two separate questions, deliberately not mixed:

  AGREEMENT  ours vs HuggingFace on the same prompts with greedy decoding.
             This is the implementation gate. If paged KV, batching, prefix
             caching or CUDA graphs have a subtle bug that the logit tests
             missed, it shows up here as diverging answers. Greedy is used so
             both sides are deterministic and comparable.

  QUALITY    ours with Qwen's recommended sampling, scored against the
             published number. This is a sanity check on the whole stack:
             tokenizer, chat template, RoPE, kernels. It cannot be exact —
             sampling adds a couple of points of variance.

Two things the Qwen docs insist on, and both matter here:

  - Thinking mode is ON by default and emits <think>...</think>. We pass
    enable_thinking=False, because thinking needs up to 38k output tokens and
    the published non-thinking numbers are what a short run can reach.
  - Do not greedy decode for the quality run. Qwen warns it causes repetition
    loops and score collapse. Greedy is fine for the agreement run, where the
    point is determinism rather than a good score.

Published references (Qwen3 technical report):
  MATH-500  87.4 non-thinking / 97.4 thinking   (instruct model)
  GSM8K     89.84 4-shot CoT                    (8B-BASE, not instruct)
"""

import argparse
import json
import os
import re
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.block_manager import build_manager_for
from engine.cuda_graph import DecodeGraphRunner
from engine.llm_engine import LLMEngine
from engine.scheduler import Scheduler
from models.qwen3 import Qwen3ForCausalLM

MODEL = "Qwen/Qwen3-8B"
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Qwen's recommended non-thinking sampling.
QWEN_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20)
GREEDY = dict(temperature=0.0, top_p=1.0, top_k=0)

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

PUBLISHED = {
    "math500": ("87.4", "non-thinking, instruct model"),
    "gsm8k": ("89.84", "4-shot CoT, 8B-BASE not instruct — expect a rough match only"),
}


# ---------------------------------------------------------------- extraction

def extract_boxed(text: str) -> str | None:
    """Last \\boxed{...}, brace-matched so nested braces survive."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def last_number(text: str) -> str | None:
    found = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return found[-1].replace(",", "") if found else None


def normalise_math(s: str) -> str:
    """Fold the harmless LaTeX variations so string compare has a chance.

    This still undercounts: "\\frac{\\sqrt{3}}{2}" and "0.866..." are equal
    mathematically and different as strings. A real harness uses sympy. Scores
    here are therefore a FLOOR, not the true score.
    """
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\(left|right|!|,|;|\s)", "", s)
    s = re.sub(r"\\(d|t)frac", r"\\frac", s)
    s = re.sub(r"\\(text|mbox|mathrm)\{([^{}]*)\}", r"\2", s)
    s = re.sub(r"\^\{?\\circ\}?", "", s)      # degrees
    s = s.replace("\\%", "").replace("%", "")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace(" ", "").replace("\\", "").rstrip(".")
    # "x=5" -> "5"
    if "=" in s and s.count("=") == 1:
        s = s.split("=")[-1]
    return s


def numeric(s: str) -> float | None:
    """Parse a number out of model output.

    The model writes money and percentages the way a human would — "\$70,000",
    "60\%" — while GSM8K gold answers are bare digits. Stripping the
    decoration first turned three of the first three "misses" into hits, so
    without it the score is an artefact of the harness, not the model.
    """
    if s is None:
        return None
    for junk in (",", "$", "\\", "%", " ", "\u00a0"):
        s = s.replace(junk, "")
    s = s.strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def is_correct(pred: str | None, gold: str, task: str) -> bool:
    if pred is None:
        return False
    if task == "gsm8k":
        p, g = numeric(pred), numeric(gold)
        return p is not None and g is not None and abs(p - g) < 1e-4
    a, b = normalise_math(pred), normalise_math(gold)
    if a == b:
        return True
    pa, pb = numeric(a), numeric(b)
    return pa is not None and pb is not None and abs(pa - pb) < 1e-6


def answer_of(text: str, task: str) -> str | None:
    boxed = extract_boxed(text)
    if boxed is not None:
        return boxed
    # Models sometimes forget the box; fall back to the last number.
    return last_number(text)


# ---------------------------------------------------------------- data

def load_task(task: str, n: int):
    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        rows = [(r["question"], r["answer"].split("####")[-1].strip()) for r in ds]
    else:
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        rows = [(r["problem"], r["answer"]) for r in ds]
    return rows[:n]


def build_prompts(tokenizer, questions):
    """Chat template with thinking disabled."""
    out = []
    for q in questions:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"{q}\n\n{INSTRUCTION}"}],
            add_generation_prompt=True, tokenize=False, enable_thinking=False,
        )
        out.append(tokenizer(text, add_special_tokens=False).input_ids)
    return out


# ---------------------------------------------------------------- runners

def run_ours(model, tokenizer, prompts, max_new, sampling, runner=None,
             num_blocks=400, max_num_seqs=32):
    manager = build_manager_for(model, num_blocks=num_blocks, block_size=256)
    scheduler = Scheduler(manager, max_num_seqs=max_num_seqs,
                          max_num_batched_tokens=8192)
    engine = LLMEngine(model, manager, scheduler, runner)
    for ids in prompts:
        engine.add_request(list(ids), max_new, tokenizer.eos_token_id, **sampling)

    t0 = time.perf_counter()
    finished = engine.run_all()
    elapsed = time.perf_counter() - t0

    finished.sort(key=lambda s: s.seq_id)
    texts, generated = [], 0
    for seq, ids in zip(finished, prompts):
        new = seq.token_ids[len(ids):]
        generated += len(new)
        texts.append(tokenizer.decode(new, skip_special_tokens=True))
    return texts, elapsed, generated


def run_hf(tokenizer, prompts, max_new, batch=8):
    """HuggingFace generate(), greedy, left-padded batches."""
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    texts = []
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        width = max(len(c) for c in chunk)
        ids = torch.tensor([[tokenizer.pad_token_id] * (width - len(c)) + c
                            for c in chunk], device="cuda")
        mask = torch.tensor([[0] * (width - len(c)) + [1] * len(c)
                             for c in chunk], device="cuda")
        with torch.inference_mode():
            out = hf.generate(ids, attention_mask=mask, max_new_tokens=max_new,
                              do_sample=False, pad_token_id=tokenizer.pad_token_id)
        for row in out[:, width:]:
            texts.append(tokenizer.decode(row, skip_special_tokens=True))
    elapsed = time.perf_counter() - t0

    del hf
    torch.cuda.empty_cache()
    return texts, elapsed


# ---------------------------------------------------------------- main

def score(texts, golds, task):
    hits = [is_correct(answer_of(t, task), g, task) for t, g in zip(texts, golds)]
    boxed = sum(extract_boxed(t) is not None for t in texts)
    return sum(hits) / len(hits), hits, boxed / len(texts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["gsm8k", "math500"], default="gsm8k")
    ap.add_argument("--n", type=int, default=200, help="problems for the quality run")
    ap.add_argument("--n-agree", type=int, default=40, help="problems for ours-vs-HF")
    ap.add_argument("--max-new", type=int, default=768)
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    model = Qwen3ForCausalLM(hf.config).to(torch.bfloat16)
    model.load_state_dict(hf.state_dict())
    model = model.cuda().eval()
    del hf
    torch.cuda.empty_cache()

    rows = load_task(args.task, max(args.n, args.n_agree))
    questions = [q for q, _ in rows]
    golds = [g for _, g in rows]
    prompts = build_prompts(tokenizer, questions)
    print(f"[task] {args.task}, {len(rows)} problems loaded, "
          f"prompt tokens {min(map(len, prompts))}-{max(map(len, prompts))}, "
          f"max_new={args.max_new}")

    results = {"task": args.task, "model": MODEL, "max_new": args.max_new}

    # --- AGREEMENT: implementation gate ---
    if not args.skip_hf:
        k = args.n_agree
        print(f"\n=== agreement: ours vs HuggingFace, greedy, {k} problems ===")
        ours_txt, ours_s, _ = run_ours(model, tokenizer, prompts[:k],
                                       args.max_new, GREEDY)
        hf_txt, hf_s = run_hf(tokenizer, prompts[:k], args.max_new)

        same_text = sum(a == b for a, b in zip(ours_txt, hf_txt))
        ans_o = [answer_of(t, args.task) for t in ours_txt]
        ans_h = [answer_of(t, args.task) for t in hf_txt]
        same_ans = sum(
            (a == b) or is_correct(a, b or "", args.task)
            for a, b in zip(ans_o, ans_h)
        )
        acc_o, _, _ = score(ours_txt, golds[:k], args.task)
        acc_h, _, _ = score(hf_txt, golds[:k], args.task)

        print(f"  identical text     : {same_text}/{k} ({same_text / k:.0%})")
        print(f"  same final answer  : {same_ans}/{k} ({same_ans / k:.0%})")
        print(f"  accuracy ours / HF : {acc_o:.1%} / {acc_h:.1%}")
        print(f"  time     ours / HF : {ours_s:.1f}s / {hf_s:.1f}s "
              f"({hf_s / ours_s:.1f}x)")
        results["agreement"] = {
            "n": k, "identical_text": same_text, "same_answer": same_ans,
            "acc_ours": round(acc_o, 4), "acc_hf": round(acc_h, 4),
            "ours_s": round(ours_s, 1), "hf_s": round(hf_s, 1),
        }

    # --- QUALITY: Qwen's sampling, vs the published number ---
    print(f"\n=== quality: ours, Qwen non-thinking sampling, {args.n} problems ===")
    runner = None  # graphs need a fixed block-table width; long gens exceed it
    texts, elapsed, generated = run_ours(model, tokenizer, prompts[:args.n],
                                         args.max_new, QWEN_SAMPLING, runner)
    acc, hits, boxed_rate = score(texts, golds[:args.n], args.task)
    ref, note = PUBLISHED[args.task]
    print(f"  accuracy           : {acc:.1%}  ({sum(hits)}/{args.n})")
    print(f"  published reference: {ref}%  ({note})")
    print(f"  answers in \\boxed{{}}: {boxed_rate:.0%}")
    print(f"  throughput         : {generated / elapsed:.0f} tok/s "
          f"({generated} tokens in {elapsed:.1f}s)")
    results["quality"] = {
        "n": args.n, "accuracy": round(acc, 4), "published": ref,
        "boxed_rate": round(boxed_rate, 4),
        "tok_s": round(generated / elapsed), "elapsed_s": round(elapsed, 1),
    }

    # Show a couple of misses; usually extraction, not reasoning.
    misses = [i for i, h in enumerate(hits) if not h][:3]
    if misses:
        print("\n  sample misses (check whether it is scoring, not reasoning):")
        for i in misses:
            print(f"    gold {golds[i]!r}  got {answer_of(texts[i], args.task)!r}")
            print(f"      ...{texts[i][-140:].strip()!r}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"intelligence_{args.task}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[saved] {path}")


if __name__ == "__main__":
    main()
