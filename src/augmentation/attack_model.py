"""Generative model for IEDExplorer report-capture attack packets.

``AttackGenerativeModel`` is fit from the profile produced by
``profile_attacks.py`` plus the real attack rows (templates). It reproduces new
attack packets by:

  1. sampling a communication channel from the empirical ``src_ip`` PMF;
  2. drawing a real attack row of that channel as a *structural scaffold*
     (this fixes the protocol structure and the report-member count, so the
     output is always a valid URCB report);
  3. resampling *which* controllable objects the report carries, per IED and
     per object type, via a first-order Markov chain over object tokens
     (preserving the CSWI/``$ST$Pos`` vs CILO/``$ST$EnaCls`` relationship and
     the real per-IED object frequencies);
  4. advancing a monotone per-stream report-sequence counter using the
     empirical increment distribution;
  5. advancing a per-stream clock using the empirical inter-arrival
     distribution, and stamping the timestamp fields accordingly.

The model only ever emits real, observed field values; it does not interpolate
(unlike SMOTE) or learn an unconstrained generator (unlike a GAN), which makes
it both protocol-faithful and robust at the small real-sample size (n=105).

No file I/O lives here so the model is unit-testable in isolation.
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Iterator, Optional

OBJECT_RE = re.compile(r"(Q\dCSWI\d|Q\dCILO\d)")
MIN_SYNTHETIC_ATTACK_GAP_SECONDS = 61.0


def _object_type(token: str) -> str:
    return "CSWI" if "CSWI" in token else "CILO"


def _pmf_sample(rng: random.Random, pmf: dict) -> str:
    """Sample a key from a {value: count} PMF."""
    keys = list(pmf.keys())
    weights = [float(pmf[k]) for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


class _StreamState:
    """Running per-stream counter + clock used during a sampling run."""

    def __init__(self, seq_value: int, clock: datetime):
        self.seq_value = seq_value
        self.clock = clock


class AttackGenerativeModel:
    def __init__(self, profile: dict, templates: list[dict]):
        self.profile = profile
        self.templates = templates
        self.constants = profile["constants"]
        self.src_ip_pmf = profile["src_ip_pmf"]
        self.ied_objects = profile["ied_objects"]
        self.ied_urcb = profile["ied_urcb"]
        self.markov_start = profile.get("object_markov_start", {})
        self.markov = profile.get("object_markov", {})
        self.stream_seq = profile["stream_seq"]
        self.stream_timing = profile["stream_timing"]

        # Group template rows by source channel so the (channel -> IED -> URCB)
        # coupling observed in the capture is preserved.
        self.templates_by_channel: dict[str, list[dict]] = defaultdict(list)
        for row in templates:
            self.templates_by_channel[(row.get("src_ip") or "").strip()].append(row)

        self._stream_state: dict[str, _StreamState] = {}
        self._control_ctl_num = 1
        self._last_attack_clock: Optional[datetime] = None

    # -- fit helpers -------------------------------------------------------
    @classmethod
    def from_files(cls, profile_path: str, templates_path: str) -> "AttackGenerativeModel":
        with open(profile_path, encoding="utf-8") as handle:
            profile = json.load(handle)
        templates = []
        with open(templates_path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    templates.append(json.loads(line))
        return cls(profile, templates)

    # -- object-sequence Markov sampling -----------------------------------
    def _sample_object_sequence(self, rng: random.Random, ied: str, obj_type: str, k: int) -> list[str]:
        vocab = self.ied_objects.get(ied, {}).get(obj_type, {})
        if not vocab:
            return []
        start = self.markov_start.get(ied, {}).get(obj_type, {}) or vocab
        transitions = self.markov.get(ied, {}).get(obj_type, {})
        tokens = [_pmf_sample(rng, start)]
        while len(tokens) < k:
            nxt = transitions.get(tokens[-1])
            tokens.append(_pmf_sample(rng, nxt) if nxt else _pmf_sample(rng, vocab))
        return tokens

    # -- per-stream counter / clock ---------------------------------------
    def _stream_state_for(self, rng: random.Random, stream: str) -> _StreamState:
        if stream not in self._stream_state:
            seq_model = self.stream_seq.get(stream, {"start_min": 1})
            time_model = self.stream_timing.get(stream, {})
            start_ts = time_model.get("start_timestamp")
            clock = datetime.fromisoformat(start_ts) if start_ts else datetime(2026, 2, 14)
            self._stream_state[stream] = _StreamState(int(seq_model.get("start_min", 1)), clock)
        return self._stream_state[stream]

    def _advance_stream(self, rng: random.Random, stream: str) -> tuple[int, datetime]:
        state = self._stream_state_for(rng, stream)
        seq_model = self.stream_seq.get(stream, {})
        inc_pmf = seq_model.get("increment_pmf", {"1": 1})
        gap_sample = self.stream_timing.get(stream, {}).get("gap_seconds_sample", [17.0])
        gap_seconds = max(float(rng.choice(gap_sample)), MIN_SYNTHETIC_ATTACK_GAP_SECONDS)
        state.seq_value += int(_pmf_sample(rng, inc_pmf))
        candidate_clock = state.clock + timedelta(seconds=gap_seconds)
        if self._last_attack_clock is not None:
            candidate_clock = max(
                candidate_clock,
                self._last_attack_clock + timedelta(seconds=MIN_SYNTHETIC_ATTACK_GAP_SECONDS),
            )
        state.clock = candidate_clock
        self._last_attack_clock = candidate_clock
        return state.seq_value, state.clock

    # -- access_result mutation -------------------------------------------
    @staticmethod
    def _mutate_access_result(
        access_raw: str,
        new_tokens: list[str],
        new_seq: int,
        new_clock: datetime,
        new_ctl_num: Optional[int] = None,
    ) -> str:
        try:
            items = json.loads(access_raw)
        except (json.JSONDecodeError, TypeError):
            return access_raw

        token_iter = iter(new_tokens)
        unsigned_index = 0
        new_utc = new_clock.replace(microsecond=0).isoformat()

        def walk(node):
            nonlocal unsigned_index
            if isinstance(node, list):
                for element in node:
                    walk(element)
            elif isinstance(node, dict):
                ntype = node.get("type")
                value = node.get("value")
                if ntype == "visible-string" and isinstance(value, str) and OBJECT_RE.search(value):
                    replacement = next(token_iter, None)
                    if replacement is not None:
                        node["value"] = OBJECT_RE.sub(replacement, value, count=1)
                elif ntype == "unsigned" and isinstance(value, int):
                    if unsigned_index == 0:
                        node["value"] = int(new_seq)
                    elif new_ctl_num is not None:
                        node["value"] = int(new_ctl_num)
                    unsigned_index += 1
                elif ntype == "utc-time" and isinstance(value, str):
                    node["value"] = new_utc
                elif ntype in ("list", "structure") or isinstance(value, list):
                    walk(value)

        walk(items)
        return json.dumps(items, separators=(",", ":"), ensure_ascii=False)

    def _advance_control_ctl_num(self) -> int:
        value = self._control_ctl_num
        self._control_ctl_num += 1
        return value

    # -- sampling ----------------------------------------------------------
    def sample_one(self, rng: random.Random) -> dict:
        channel = _pmf_sample(rng, self.src_ip_pmf)
        pool = self.templates_by_channel.get(channel) or self.templates
        template = dict(rng.choice(pool))
        access_raw = template.get("access_result") or ""

        # Group the template's object slots by type, preserving order.
        slot_tokens = OBJECT_RE.findall(access_raw)
        ied_match = re.search(r"([A-Z0-9]+_CTRL)/", access_raw)
        ied = ied_match.group(1) if ied_match else ""

        counts = {"CSWI": sum(1 for t in slot_tokens if _object_type(t) == "CSWI"),
                  "CILO": sum(1 for t in slot_tokens if _object_type(t) == "CILO")}
        sampled = {
            "CSWI": iter(self._sample_object_sequence(rng, ied, "CSWI", counts["CSWI"])),
            "CILO": iter(self._sample_object_sequence(rng, ied, "CILO", counts["CILO"])),
        }
        # Re-emit replacements in the original slot order, per type.
        new_tokens = []
        for token in slot_tokens:
            replacement = next(sampled[_object_type(token)], token)
            new_tokens.append(replacement)

        stream = (template.get("stream_id") or "").strip()
        new_seq, new_clock = self._advance_stream(rng, stream)
        new_ctl_num = self._advance_control_ctl_num()

        row = dict(template)
        row["access_result"] = self._mutate_access_result(
            access_raw, new_tokens, new_seq, new_clock, new_ctl_num
        )
        row["timestamp"] = new_clock.isoformat()
        # Ensure fingerprint constants are exactly the real values.
        for field, value in self.constants.items():
            row[field] = value
        return row

    def sample(self, n: int, rng: random.Random) -> Iterator[dict]:
        self._stream_state = {}
        self._control_ctl_num = 1
        self._last_attack_clock = None
        for _ in range(n):
            yield self.sample_one(rng)
