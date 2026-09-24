"""On-disk store primitives for the paged-cache save/restore surface (kv-persistence P1).

Pure, engine-agnostic mechanics ported from the proven P0 oracle
`~/issues/local-llm/hosting/kv-persistence/probes/probe_lib.py` (`capture_store` / `restore_store`,
all-probes-PASS 2026-09-22): the store file set and write order (pages.bin + chain.json + stash-*.bin
oldest->newest, per-file sha256 over ALL staged files, meta.json LAST, temp dir + atomic rename), the
root->leaf chain ordering, the deepest-K stash budget walk, dual-device syncs, and the validate-first
(fail-closed) pin/digest checks. `Generator.save_state()` / `Generator.restore_state()` compose these;
the sequencing that makes the capture Q1-safe (enumerate defrag-invariant metadata, then resolve
`page.page_index` fresh at copy time) lives on the Generator itself.

Nothing here touches the page table or the recurrent cache: it is byte/file/hash plumbing only.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import time

import torch


# Deepest-K stash budget, plan §2 (512 MiB). `save_state` carries no budget kwarg — the contract's
# signature is exactly one positional store path — so the default lives at module level.
STASH_BUDGET_DEFAULT = 512 * 1024 ** 2


def sha256_file(path: str) -> str:
    """Hex sha256 of a file's bytes (Layer-A integrity basis)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_version() -> str:
    """Package version via importlib.metadata. This fork exposes no `exllamav3.__version__` attribute
    (empirical 2026-09-22), so metadata is the only source; 'unknown' is stamped loudly rather than
    crashing if the dist-info is missing."""
    try:
        from importlib.metadata import version as _pkg_version
        return _pkg_version("exllamav3")
    except Exception:
        return "unknown"


def engine_fork_revision() -> str:
    """Fork git revision, provenanced. NEVER stamp a parent repo's HEAD: a venv-installed copy of this
    package under a tabbyAPI checkout resolves `git rev-parse` to the TABBYAPI repo (empirically
    verified), which would stamp a meaningless revision. Require git's toplevel to BE the exllamav3
    source root; else fall back to the EXL3_KVSAVE_REV env pin (what a venv copy must carry); else
    raise — a capture whose revision cannot be proven is worthless for forensics, so it fails loudly."""
    import exllamav3
    root = os.path.dirname(os.path.dirname(os.path.abspath(exllamav3.__file__)))
    try:
        top = subprocess.run(["git", "-C", root, "rev-parse", "--show-toplevel"],
                             capture_output = True, text = True, timeout = 5).stdout.strip()
        if top and os.path.realpath(top) == os.path.realpath(root):
            rev = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                                 capture_output = True, text = True, timeout = 5).stdout.strip()
            if rev:
                return rev
    except Exception:
        pass
    rev = os.environ.get("EXL3_KVSAVE_REV", "").strip()
    if rev:
        return rev
    raise RuntimeError(f"cannot prove fork revision: git toplevel at {root} is not its own repo "
                       "(a venv copy resolves to a parent repo) and EXL3_KVSAVE_REV is unset")


def expect_pin(label: str, expected, actual):
    """Raise with expected/actual on a pin mismatch (restore-side validation, fail-closed)."""
    if expected != actual:
        raise RuntimeError(f"pin {label} mismatch: expected={expected} actual={actual}")


def device_set(tensors) -> list:
    """Distinct devices among the paged cache tensors, ordered by device index. The [24,8] split puts
    cache tensors on both cards, so this is never just the current device."""
    return sorted({t.device for t in tensors if t is not None}, key = lambda d: d.index if d.index is not None else -1)


def sync_devices(devices):
    """torch.cuda.synchronize() every device touched by the cache — both cards, never just current."""
    for dev in devices:
        torch.cuda.synchronize(dev)


def paged_tensors(cache) -> list:
    """Paged cache tensors, the same collection the CPU page tier builds segments from
    (generator/cpu_cache.py via cache.get_all_tensors()). Non-attention layers contribute None slots
    (cache/fp16.py:31-33, cache/quant.py:62-66), hence the filter."""
    return [t for t in cache.get_all_tensors() if t is not None]


def tensor_manifest(cache) -> list:
    """Per-tensor page-slice (shape, dtype) fingerprint: the restore-side shape/dtype gate."""
    return [[list(t[0].shape), str(t.dtype).replace("torch.", "")] for t in paged_tensors(cache)]


def k_bits_of(cache) -> int:
    """k_bits of the first quant cache layer (0 when the cache is fp16)."""
    for l in cache.layers.values():
        return int(getattr(l, "k_bits", 0))
    return 0


def v_bits_of(cache) -> int:
    """v_bits of the first quant cache layer (0 when the cache is fp16)."""
    for l in cache.layers.values():
        return int(getattr(l, "v_bits", 0))
    return 0


def cache_mode_of(cache) -> str:
    return f"{k_bits_of(cache)},{v_bits_of(cache)}"


def derive_gpu_split(cache) -> list:
    """gpu_split pin, derived from live objects.

    The engine does NOT retain the load-time placement: `Model.load(use_per_device = [...])` converts
    GiB to bytes at model.py:414 and hands the list to the loaders, where `device_budget` is a local of
    `_load_autosplit` (model_ls.py:117) — nothing lands on the model object. So the pin is derived as
    the per-device GiB footprint of the paged cache tensors (sum of numel*element_size per device
    index, ordered by index, rounded to 2 decimals), which mirrors the probe's list-of-GiB-floats
    format while being recoverable on both sides of a round trip:

      * save stamps `derive_gpu_split(cache)` and restore re-derives it from ITS cache and compares —
        identical placement validates, so a store is never rejected for a field it cannot recover;
      * a different split moves whole cache layers between cards, changing the per-device byte sums,
        so a cross-layout restore still rejects (which is the point of the pin).

    Limits, stated plainly: this fingerprints the CACHE placement, not the requested VRAM budgets, so
    two splits that happen to park identical byte totals on each card would compare equal, and a
    placement that drifts between two loads of the same split (autosplit is size-driven, not
    free-RAM-driven, when use_per_device is given) would compare unequal and reject fail-closed. Both
    are acceptable for v1: rejection is loud and cold, never silent.
    """
    per_device = {}
    for t in paged_tensors(cache):
        per_device[t.device] = per_device.get(t.device, 0) + int(t.numel()) * int(t.element_size())
    if not per_device:
        return []
    ordered = sorted(per_device, key = lambda d: d.index if d.index is not None else -1)
    return [round(per_device[dev] / 1024 ** 3, 2) for dev in ordered]


def chain_order(pages):
    """Order complete content-hashed pages root->leaf. BFS from roots (prev_hash is None) sorted by
    access_serial, children likewise, so a chain is emitted parent before child and sibling branches
    oldest-first. Returns (ordered hashes, gaps, by_hash); a page whose prev_hash names no captured
    page is a gap — one gap makes everything above it non-resumable (pagetable.py:659)."""
    by_hash = {p.phash: p for p in pages}
    children = {p.phash: [] for p in pages}
    roots, gaps = [], []
    for p in pages:
        if p.prev_hash is None:
            roots.append(p.phash)
        elif p.prev_hash in children:
            children[p.prev_hash].append(p.phash)
        else:
            gaps.append(p.phash)
    order = []
    stack = sorted(roots, key = lambda h: by_hash[h].access_serial)
    while stack:
        h = stack.pop(0)
        order.append(h)
        stack.extend(sorted(children[h], key = lambda c: by_hash[c].access_serial))
    return order, gaps, by_hash


def select_stashes(items, budget: int = STASH_BUDGET_DEFAULT) -> list:
    """Deepest-K stash selection. `items` is `list(recurrent_cache.items())`, an OrderedDict in
    oldest->newest order. Walk newest-first keeping stashes while the budget allows, then reverse so
    the file order written is oldest->newest — restore's `put()` loop then re-establishes LRU recency
    in true recency order. The first (newest) stash is always kept even if it alone exceeds the
    budget, matching the probe (`if keep and total + sz > budget`)."""
    keep = []
    total = 0
    for k, v in reversed(items):
        sz = int(v["checkpoint_size"])
        if keep and total + sz > budget:
            break
        keep.append((k, v))
        total += sz
    keep.reverse()
    return keep


def tensor_bytes(t: torch.Tensor) -> bytes:
    """Raw little-endian bytes of a tensor (contiguous uint8 view; dtype-agnostic, covers the bf16/half
    dtypes numpy cannot hold natively)."""
    return t.contiguous().view(torch.uint8).cpu().numpy().tobytes()


def bytes_tensor(raw, dtype_str: str, shape):
    """Rebuild a CPU tensor from raw bytes plus the dtype/shape recorded at capture."""
    dtype = getattr(torch, dtype_str)
    buf = torch.frombuffer(bytearray(raw), dtype = torch.uint8)
    return buf.view(dtype).view(shape)


def stash_blob(state: dict):
    """Serialise one stashed recurrent state: every (layer_idx, instance) keyed entry, in dict order,
    as raw bytes plus a positional descriptor [key, nbytes, shape, dtype, kind]. `kind` is per key
    (GDN stashes a (recurrent_state, conv_state) PAIR under one key; other module types stash a bare
    TENSOR), and must be remembered per key on restore — not taken from the last descriptor."""
    tbuf = bytearray()
    tdesc = []
    for key in [kk for kk in state.keys() if isinstance(kk, tuple)]:
        pair = state[key]
        tensors_pair = pair if isinstance(pair, (tuple, list)) else [pair]
        kind = "pair" if isinstance(pair, (tuple, list)) else "tensor"
        for t in tensors_pair:
            raw = tensor_bytes(t)
            tdesc.append([[int(x) for x in key], len(raw), list(t.shape),
                          str(t.dtype).replace("torch.", ""), kind])
            tbuf += raw
    return bytes(tbuf), tdesc


def stash_state_from_blob(raw: bytes, tdesc: list):
    """Rebuild a stashed state dict from its bytes + descriptors (inverse of `stash_blob`)."""
    state = {}
    pos = 0
    order_keys = []
    key_kind = {}
    for td in tdesc:
        key, nbytes, shape, dtype_str, kind = tuple(td[0]), td[1], td[2], td[3], td[4]
        state.setdefault(key, []).append(bytes_tensor(raw[pos:pos + nbytes], dtype_str, shape))
        pos += nbytes
        if key not in order_keys:
            order_keys.append(key)
            key_kind[key] = kind
    for key in order_keys:
        state[key] = tuple(state[key]) if key_kind[key] == "pair" else state[key][0]
    return state


class PreStashed:
    """Wrap an already-rebuilt GDN state dict so `RecurrentCache.put()` stores it verbatim: put() calls
    `state.stash()` (cache/recurrent.py:59) and takes `checkpoint_size` from the result, so the shim
    hands back the same dict instead of re-stashing from a state slot."""

    def __init__(self, d: dict):
        self.d = d

    def stash(self) -> dict:
        return self.d


def digest_dir(stage_dir: str) -> dict:
    """Per-file {sha256, size} over every staged file. MUST be computed before meta.json is written,
    so meta itself is not in its own digest map (the store's own integrity is anchored by the payload
    digests plus meta-LAST ordering)."""
    files = {}
    for fn in sorted(os.listdir(stage_dir)):
        fp = os.path.join(stage_dir, fn)
        files[fn] = {"sha256": sha256_file(fp), "size": os.path.getsize(fp)}
    return files


def validate_store_dir(store_dir: str, files: dict):
    """Layer-A gate: size+sha256 of every recorded file, before any payload is read or mutated."""
    for fn, rec in files.items():
        fp = os.path.join(store_dir, fn)
        if not os.path.exists(fp) or os.path.getsize(fp) != rec["size"] or sha256_file(fp) != rec["sha256"]:
            raise RuntimeError(f"digest/size validation failed for {fn}")


def stage_atomic_rename(stage_dir: str, final_dir: str):
    """G6 atomicity: publish the staged dir with a rename, keeping any previous set intact until the
    new one has landed, and only then dropping the old one."""
    old = None
    if os.path.exists(final_dir):
        old = final_dir + ".old-" + str(os.getpid())
        os.rename(final_dir, old)
    os.rename(stage_dir, final_dir)
    if old is not None:
        shutil.rmtree(old, ignore_errors = True)


def now_stamp() -> float:
    return time.time()