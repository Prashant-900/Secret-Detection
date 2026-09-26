"""llm_backend.py - one small interface over vLLM (GPU) and a mock (CPU, for dry runs).

    be = make_backend(backend="vllm", model="Qwen/Qwen2.5-Coder-32B-Instruct", tp=1)
    texts = be.chat([(system, user), ...], max_tokens=1800, temperature=0.9, top_p=0.95, seed=0)
    p_yes = be.yes_no([(system, user), ...])          # P("YES") vs P("NO") from first-token logprobs

Sizing for ~100 GB VRAM:
  * one 80-96 GB GPU : 32B bf16 (~65 GB) or 70B FP8 (pass quantization="fp8")
  * several GPUs     : set tp=N (tensor parallel) and a larger model
Use a DIFFERENT model family for the judge/baseline than for generation, so the LLM baseline is not
graded on text written by its own family.
"""
import hashlib
import math
import re


class MockBackend:
    """Deterministic stand-in so the whole pipeline can be dry-run on CPU."""

    def __init__(self, responder=None):
        self.responder = responder or (lambda system, user: "")

    def chat(self, batch, **kw):
        return [self.responder(s, u) for s, u in batch]

    def yes_no(self, batch, **kw):
        out = []
        for _, u in batch:
            m = re.search(r">> (.*)", u)
            line = m.group(1) if m else u
            looks = bool(re.search(r"[A-Za-z0-9+/_\-]{16,}", line)) and not re.search(r"\$\{|var\.|\{\{|changeme|<", line)
            h = int(hashlib.sha1(u.encode()).hexdigest(), 16) % 100 / 1000.0
            out.append(0.85 + h if looks else 0.1 + h)
        return out


class VLLMBackend:
    def __init__(self, model, tp=1, max_len=8192, gpu_util=0.90, dtype="bfloat16", quantization=None):
        from transformers import AutoTokenizer
        from vllm import LLM
        kw = dict(model=model, tensor_parallel_size=tp, dtype=dtype, max_model_len=max_len,
                  gpu_memory_utilization=gpu_util, enable_prefix_caching=True)
        if quantization:
            kw["quantization"] = quantization
        self.llm = LLM(**kw)
        self.tok = AutoTokenizer.from_pretrained(model)

    def _prompt(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def chat(self, batch, max_tokens=1800, temperature=0.9, top_p=0.95, seed=0, **kw):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, seed=seed)
        outs = self.llm.generate([self._prompt(s, u) for s, u in batch], sp)
        return [o.outputs[0].text for o in outs]

    def yes_no(self, batch, **kw):
        from vllm import SamplingParams
        sp = SamplingParams(max_tokens=1, temperature=0.0, logprobs=20)
        outs = self.llm.generate([self._prompt(s, u) for s, u in batch], sp)
        res = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            yes = no = -1e9
            for tid, l in lp.items():
                tok = (l.decoded_token or "").strip().lower()
                if tok == "yes":
                    yes = max(yes, l.logprob)
                elif tok == "no":
                    no = max(no, l.logprob)
            if yes < -1e8 and no < -1e8:
                res.append(0.5)
            else:
                m = max(yes, no)
                ey, en = math.exp(yes - m), math.exp(no - m)
                res.append(ey / (ey + en))
        return res


def add_backend_args(ap):
    ap.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-32B-Instruct")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (number of GPUs for one model)")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--quantization", default=None, help="e.g. fp8 for a ~70B model on one 96 GB GPU")


def make_backend(a, responder=None):
    if a.backend == "mock":
        return MockBackend(responder)
    return VLLMBackend(a.model, tp=a.tp, max_len=a.max_len, quantization=a.quantization)
