"""Local Qwen3-8B answerer with interpretability probes.

Implements the Answerer protocol against a local HF causal LM and records
generation logprobs, claim extraction, P(IK), P(True) and semantic-entropy
probes through a ProbeRecorder. enable_thinking=False keeps latency within
the cost dimensions tracked by the pipeline.
"""

from __future__ import annotations

import json
import math
import re
import time
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .generation import Answerer
from .schemas import Answer, Claim, Evidence, Question

SYS_PROMPT = (
    "You are a careful materials-science research assistant. Answer only from "
    "the information you actually have; do not invent numeric values, and keep "
    "the answer to at most three sentences."
)
REVISION_PROMPT = (
    "Rewrite the draft answer using only claims directly supported by the "
    "evidence below. Remove unsupported examples, numbers, categories, and "
    "inferences. Keep the user's language, answer the question briefly, and "
    "say that the evidence is insufficient only when no supported answer can "
    "be formed. Cite evidence ids in brackets.\n\n"
    "Evidence:\n{evidence}\n\nQuestion: {question}\n\n"
    "Draft answer:\n{draft}\n\nRevised answer:"
)
CLAIM_PROMPT = (
    "Extract the atomic factual claims of the following answer as JSON: a list "
    "of objects with keys \"key\" (short property name, e.g. band_gap), "
    "\"value\" (exact value with unit if any) and \"source_refs\" (list of "
    "evidence ids used, empty list if none). Reply with the JSON list only.\n\n"
    "Answer:\n{text}"
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


class QwenAnswerer(Answerer):
    """Local LLM answerer; probe capture is optional but on by default."""

    def __init__(
        self,
        model_path: str,
        recorder: Any = None,
        max_new_tokens: int = 200,
        entropy_samples: int = 4,
        entropy_temperature: float = 0.8,
        hidden_states: bool = False,
        device: str = "cuda",
    ) -> None:
        self.model_path = model_path
        self.recorder = recorder
        self.max_new_tokens = max_new_tokens
        self.entropy_samples = entropy_samples
        self.entropy_temperature = entropy_temperature
        self.hidden_states = hidden_states
        self.device = device
        self._tok = AutoTokenizer.from_pretrained(model_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device)
        self._model.eval()
        self._hs_cache: dict[int, torch.Tensor] = {}
        self._hook_handles: list = []
        if self.hidden_states:
            self._register_hooks()

    def _register_hooks(self) -> None:
        for name, mod in self._model.named_modules():
            if name.endswith("layers"):
                self._hook_handles.append(
                    mod.register_forward_hook(self._hs_hook))

    # ---------- Answerer protocol ----------

    def generate(self, question: Question,
                 evidence: Sequence[Evidence] = ()) -> Answer:
        mode = "grounded" if evidence else "parametric"
        if evidence:
            ctx = "\n\n".join(f"[{e.evidence_id}] {e.text}" for e in evidence)
            user = (f"Evidence:\n{ctx}\n\nQuestion: {question.text}\n"
                    "Answer using the evidence, and cite evidence ids in "
                    "brackets like [id].")
        else:
            user = f"Question: {question.text}"
        text, gen_probe = self._chat(user, greedy=True)
        if self.recorder is not None:
            self.recorder.record(question.question_id or "?", "generation",
                                 {"mode": mode, **gen_probe})
        claims = self._extract_claims(question, text, mode)
        return Answer(text=text, claims=tuple(claims))

    def revise_grounded(self, question: Question, draft: Answer,
                        evidence: Sequence[Evidence], verification) -> Answer:
        """Remove unsupported draft claims before the final gate decision."""
        context = "\n\n".join(
            f"[{item.evidence_id}] {item.text}" for item in evidence
        )
        unsupported = ", ".join(verification.unsupported_claims) or "the unsupported parts"
        prompt = REVISION_PROMPT.format(
            evidence=context,
            question=question.text,
            draft=(
                f"{draft.text}\n\nUnsupported claim keys to remove: {unsupported}"
            ),
        )
        text, probe = self._chat(prompt, greedy=True)
        if self.recorder is not None:
            self.recorder.record(
                question.question_id or "?",
                "grounded_revision",
                {"text": text, **probe, "removed_claims": list(verification.unsupported_claims)},
            )
        claims = self._extract_claims(question, text, "grounded_revision")
        return Answer(text=text, claims=tuple(claims))

    # ---------- probes ----------

    def probe_self_knowledge(self, question: Question) -> dict:
        """Kadavath-style P(IK): P(Yes) vs P(No) on a knowledge prompt."""
        prompt = (f"Question: {question.text}\n\n"
                  "Do you know the answer to this question? "
                  "Reply with a single word: Yes or No.\nAnswer:")
        p_yes, p_no = self._score_first_token(prompt, ("Yes", "No"))
        p_ik = p_yes / max(p_yes + p_no, 1e-9)
        out = {"p_yes": round(p_yes, 6), "p_no": round(p_no, 6),
               "p_ik": round(p_ik, 6), "prompt": prompt}
        if self.recorder is not None:
            self.recorder.record(question.question_id or "?", "p_ik", out)
        return out

    def probe_p_true(self, question: Question) -> dict:
        """P(True): score the greedy answer as True/False on a judge prompt."""
        greedy, _ = self._chat(f"Question: {question.text}", greedy=True)
        prompt = (f"Question: {question.text}\nProposed answer: {greedy}\n\n"
                  "Is the proposed answer correct? Reply True or False.\nAnswer:")
        p_t, p_f = self._score_first_token(prompt, ("True", "False"))
        out = {"p_true": round(p_t / max(p_t + p_f, 1e-9), 6),
               "greedy_answer": greedy, "prompt": prompt}
        if self.recorder is not None:
            self.recorder.record(question.question_id or "?", "p_true", out)
        return out

    def probe_semantic_entropy(self, question: Question) -> dict:
        """Sample k answers, cluster by normalized text, report entropy."""
        samples = []
        for _ in range(self.entropy_samples):
            text, _ = self._chat(f"Question: {question.text}", greedy=False,
                                 temperature=self.entropy_temperature)
            samples.append(text)
        clusters: dict[str, int] = {}
        for s in samples:
            clusters[_norm(s)] = clusters.get(_norm(s), 0) + 1
        probs = [c / len(samples) for c in clusters.values()]
        entropy = -sum(p * math.log(p) for p in probs) / math.log(
            max(len(samples), 2))  # normalized to [0,1]
        out = {"entropy": round(entropy, 6), "n_samples": len(samples),
               "clusters": len(clusters), "samples": samples}
        if self.recorder is not None:
            self.recorder.record(question.question_id or "?",
                                 "semantic_entropy", out)
        return out

    # ---------- internals ----------

    def _chat(self, user: str, greedy: bool,
              temperature: float = 0.7) -> tuple[str, dict]:
        msgs = [{"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user}]
        text = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        ids = self._tok(text, return_tensors="pt").to(self.device)
        t0 = time.time()
        kwargs: dict = dict(max_new_tokens=self.max_new_tokens,
                            return_dict_in_generate=True, output_scores=True,
                            pad_token_id=self._tok.pad_token_id or
                            self._tok.eos_token_id)
        if greedy:
            kwargs["do_sample"] = False
        else:
            kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
        with torch.no_grad():
            out = self._model.generate(**ids, **kwargs)
        latency = (time.time() - t0) * 1000.0
        gen_ids = out.sequences[0][ids["input_ids"].shape[1]:]
        text_out = self._tok.decode(gen_ids, skip_special_tokens=True).strip()
        lps = []
        for i, scores in enumerate(out.scores):
            step = scores[0].float()
            step = torch.log_softmax(step, dim=-1)
            lps.append(step[gen_ids[i]].item())
        probe = {
            "text": text_out,
            "mean_logprob": round(sum(lps) / max(len(lps), 1), 6),
            "min_logprob": round(min(lps) if lps else 0.0, 6),
            "n_tokens": len(lps),
            "latency_ms": round(latency, 1),
        }
        if self.hidden_states and self.recorder is not None:
            layers = sorted(self._hs_cache)
            payload = {"mode": "grounded" if "Evidence" in user else "parametric",
                       "layers": layers,
                       "mean_pooled": {str(L): [round(v, 5) for v in
                                                self._hs_cache[L].tolist()]
                                       for L in layers}}
            self.recorder.record("?", "hidden_states", payload)
            self._hs_cache.clear()
        return text_out, probe

    def _hs_hook(self, module, args, output):
        # output[0]: (batch, seq, hidden); mean-pool over seq, store by layer idx
        h = output[0][0].float().mean(dim=0).cpu()
        idx = len(self._hs_cache)
        self._hs_cache[idx] = h

    def _score_first_token(self, prompt_text: str,
                           candidates: tuple[str, ...]) -> tuple[float, float]:
        text = self._tok.apply_chat_template(
            [{"role": "user", "content": prompt_text}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        ids = self._tok(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self._model(**ids).logits[0, -1].float()
        logp = torch.log_softmax(logits, dim=-1)
        vals = [logp[self._tok(c, add_special_tokens=False)["input_ids"][0]]
                .item() for c in candidates]
        m = max(vals)
        exps = [math.exp(v - m) for v in vals]
        z = sum(exps)
        return exps[0] / z, exps[1] / z

    def _extract_claims(self, question: Question, answer_text: str,
                        mode: str) -> list[Claim]:
        prompt = CLAIM_PROMPT.format(text=answer_text)
        raw, _ = self._chat(prompt, greedy=True, )
        raw = raw.strip()
        parse_ok = False
        claims: list[Claim] = []
        try:
            m = re.search(r"\[.*\]", raw, re.S)
            data = json.loads(m.group(0) if m else raw)
            if isinstance(data, list):
                for item in data:
                    key = str(item.get("key", "")).strip()
                    value = str(item.get("value", "")).strip()
                    if key and value:
                        refs = tuple(str(r) for r in
                                     item.get("source_refs", []) or ())
                        claims.append(Claim(key=key, value=value,
                                            source_refs=refs))
                parse_ok = bool(claims)
        except Exception:  # noqa: BLE001
            parse_ok = False
        if self.recorder is not None:
            self.recorder.record(
                question.question_id or "?", "claim_extraction",
                {"mode": mode,
                 "claims": [{"key": c.key, "value": c.value,
                             "source_refs": list(c.source_refs)}
                            for c in claims],
                 "parse_ok": parse_ok})
        return claims
