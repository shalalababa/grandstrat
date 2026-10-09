"""Named, counter-based RNG streams for grandstrat.

Design §4 names six streams: `map`, `spread`, `combat` (world-keyed),
`barbarians`, `diplomacy`, and one `agent_i` per civilisation.  Every draw is
a pure function of (stream_seed, key_tuple); there is NO mutable generator
state.  Consequences:

* Same master seed + same actions => byte-identical trajectories and records.
* Per-stream seed overrides change exactly one stream (pin-by-freeze) without
  touching the others; overrides are plumbed through the public env/CLI API.
* Common-random-number pairing: two arms that draw from the same stream with
  the same keys see the same numbers regardless of execution order.

WORLD-KEYED KEYS: every combat key is
(purpose, turn, attacker_id, defender_id, roll_index) -- world identity only,
never a path counter (fight index, action index, call order).  Two identical
fights at the same world position draw identical values from different
lineages; two different fights in the same turn draw different values.

Integer implementation only; float uniform = int / 2**64 (exact, deterministic
across platforms; no libc, no Python-hash dependence).
"""

from __future__ import annotations

from typing import Hashable, Sequence

MASK64 = (1 << 64) - 1
GOLDEN = 0x9E3779B97F4A7C15

# Base streams, design §4, plus the `belief` stream (particle-filter
# belief draws; derived from the master seed, never drawn during play).
# agent_i streams are derived (one per civ id).
STREAM_NAMES = ("map", "spread", "combat", "barbarians", "diplomacy", "belief")

# Per-stream 64-bit salts (arbitrary odd constants, ASSUMED; part of the
# pinned API -- changing them changes all trajectories).
_STREAM_SALT = {
    "map": 0x1D8E4F27B6C3A509,
    "spread": 0x9F27D4C1B8A63E75,
    "combat": 0xB73E29C64D10F8A3,
    "barbarians": 0x2FA9C81E57B40D63,
    "diplomacy": 0x4C8A1F6E93D2B705,
    "belief": 0x6F2D5B84A1C7E394,
}
_AGENT_SALT = 0x7C3A9F51E2B84D60  # ASSUMED: salt for agent_i streams


def _splitmix64(x: int) -> int:
    """One splitmix64 mixing step; bijective on 64-bit words."""
    x = (x + GOLDEN) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def _key_word(k: Hashable, acc: int) -> int:
    """Fold one hashable key element into accumulator acc."""
    if isinstance(k, bool):
        return _splitmix64(acc ^ int(k))
    if isinstance(k, int):
        return _splitmix64(acc ^ (k & MASK64))
    if isinstance(k, str):
        b = k.encode("utf-8")
        for i in range(0, len(b), 8):
            chunk = b[i : i + 8]
            acc = _splitmix64(acc ^ (int.from_bytes(chunk, "little") & MASK64))
        return _splitmix64(acc ^ (len(b) + 0xABCD))
    if isinstance(k, (tuple, list)):
        for e in k:
            acc = _key_word(e, acc)
        return _splitmix64(acc ^ (len(k) + 0x77))
    if k is None:
        return _splitmix64(acc ^ 0x5EED_0000)
    if isinstance(k, float):
        # Exact folding of the IEEE bits (draws should key on ints/strs; this
        # exists so floats in keys are still deterministic, never hash-based).
        import struct

        (bits,) = struct.unpack("<Q", struct.pack("<d", float(k)))
        return _splitmix64(acc ^ bits)
    raise TypeError(f"unsupported key element type: {type(k)!r}")


def stream_seed(master_seed: int, stream: str) -> int:
    """Derive a stream's 64-bit seed from the master seed. Pure."""
    if stream.startswith("agent_"):
        idx = int(stream[len("agent_") :])
        return _splitmix64(((master_seed & MASK64) ^ _AGENT_SALT) ^ _splitmix64(idx))
    if stream not in _STREAM_SALT:
        raise KeyError(f"unknown stream {stream!r}; valid: {STREAM_NAMES} + agent_i")
    return _splitmix64((master_seed & MASK64) ^ _STREAM_SALT[stream])


def draw_u64(sseed: int, key: Sequence[Hashable]) -> int:
    """The primitive: one 64-bit draw from a keyed counter stream. Pure."""
    acc = sseed & MASK64
    for k in key:
        acc = _key_word(k, acc)
    return _splitmix64(acc)


class Stream:
    """A named, counter-based stream. Stateless; cheap to create and copy.

    `counter` (optional) is a dict that records draw counts per stream name;
    Rng.stream() passes its own draw_counts so BOTH draw paths
    (Rng.uniform and Stream.uniform) are counted uniformly."""
    __slots__ = ("name", "seed", "_counter")

    def __init__(self, name: str, seed: int, counter: dict | None = None):
        self.name = name  # any string; "agent_i" is valid
        self.seed = int(seed) & MASK64
        self._counter = counter

    def _count(self):
        if self._counter is not None:
            self._counter[self.name] = self._counter.get(self.name, 0) + 1

    def u64(self, *key: Hashable) -> int:
        self._count()
        return draw_u64(self.seed, key)

    def uniform(self, *key: Hashable) -> float:
        """Uniform float in [0, 1)."""
        self._count()
        return draw_u64(self.seed, key) / (1 << 64)

    def chance(self, p: float, *key: Hashable) -> bool:
        """Bernoulli(p). p<=0 -> False (no draw), p>=1 -> True."""
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        self._count()
        return draw_u64(self.seed, key) / (1 << 64) < p

    def randint(self, lo: int, hi: int, *key: Hashable) -> int:
        """Uniform int in [lo, hi] inclusive."""
        if hi < lo:
            raise ValueError(f"randint empty range [{lo}, {hi}]")
        self._count()
        return lo + (draw_u64(self.seed, key) % (hi - lo + 1))

    def choice(self, seq: Sequence, *key: Hashable):
        if not seq:
            raise ValueError("choice from empty sequence")
        return seq[self.randint(0, len(seq) - 1, *key)]

    def shuffle(self, items: Sequence, *key: Hashable) -> list:
        """Return a new shuffled list (Fisher-Yates keyed by position)."""
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.randint(0, i, *key, "shuffle", i)
            out[i], out[j] = out[j], out[i]
        return out

    def weighted_choice(self, weighted_pairs: Sequence[tuple], *key: Hashable):
        """weighted_pairs: [(item, weight), ...]; weight >= 0. Returns item."""
        total = sum(w for _, w in weighted_pairs)
        if total <= 0:
            raise ValueError("weighted_choice: total weight <= 0")
        x = self.uniform(*key) * total
        acc = 0.0
        for item, w in weighted_pairs:
            acc += w
            if x < acc:
                return item
        return weighted_pairs[-1][0]


class Rng:
    """Bundle of named streams for one run.

    seed             -- master seed; per-stream seeds derive from it.
    stream_overrides -- optional {stream_name: seed}; re-seed exactly one
                        stream for pin-by-freeze / CRN experiments.  Plumbed
                        through GrandStratEnv.__init__ and `gs play
                        --stream-seed name=seed` so the protocol is runnable
                        through the public API.
    """

    __slots__ = ("seed", "_overrides", "_cache", "draw_counts")

    def __init__(self, seed: int, stream_overrides: dict | None = None):
        self.seed = int(seed)
        self._overrides = dict(stream_overrides or {})
        for name in self._overrides:
            stream_seed(0, name)  # validates the name
        self._cache: dict[str, Stream] = {}
        # Per-stream draw counters: diagnostic only (reported in game record
        # footers so "this stream is inert at defaults" is always checkable).
        self.draw_counts: dict[str, int] = {}

    def stream(self, name: str) -> Stream:
        s = self._cache.get(name)
        if s is None:
            s = Stream(name, self._effective_seed(name), self.draw_counts)
            self._cache[name] = s
        return s

    def _effective_seed(self, name: str) -> int:
        if name in self._overrides:
            return int(self._overrides[name]) & MASK64
        return stream_seed(self.seed, name)

    def uniform(self, stream: str, *key: Hashable) -> float:
        self.draw_counts[stream] = self.draw_counts.get(stream, 0) + 1
        return draw_u64(self._effective_seed(stream), key) / (1 << 64)

    def chance(self, stream: str, p: float, *key: Hashable) -> bool:
        if p <= 0.0:
            return False
        if p >= 1.0:
            self.draw_counts[stream] = self.draw_counts.get(stream, 0) + 1
            return True
        return self.uniform(stream, *key) < p

    def randint(self, stream: str, lo: int, hi: int, *key: Hashable) -> int:
        self.draw_counts[stream] = self.draw_counts.get(stream, 0) + 1
        return lo + (draw_u64(self._effective_seed(stream), key) % (hi - lo + 1))

    def clone(self) -> "Rng":
        """Counter-based streams are stateless, so clone is trivially exact."""
        r = Rng(self.seed, dict(self._overrides))
        return r

    def __eq__(self, other) -> bool:
        if not isinstance(other, Rng):
            return NotImplemented
        return self.seed == other.seed and self._overrides == other._overrides

    def __repr__(self) -> str:
        return f"Rng(seed={self.seed}, overrides={self._overrides})"
