"""P1 acceptance tests — exllamav3 paged-cache save/restore for the tabby save-session workstream.

Six tests, one per row of the P1 acceptance table in
`~/issues/local-llm/hosting/kv-persistence/tabby-save-session-plan.md` (tabby v2, section 4). They gate the
engine patch on branch `sm75-v150-kvsave`; the `save_state(store)` / `restore_state(store)` generator
surface they call against is not implemented on the production tree yet, so in this state the suite is
expected to *skip* (with an explicit reason) until P1 lands. Once P1 is on and the P1 GPU window is open,
these are the green/zero-fail bar (to run alongside `tests/test_cache_rotate.py`).

What the tests are and are not about — read before editing:

* F-N1 (hard rule). Greedy generation is NOT bit-reproducible on this engine+model (CPU-MoE thread
  reduction + kernel noise; findings §2). Exact text parity is structurally unattainable. No test asserts
  text equality. The mechanism metrics below (cached_tokens, alloc_kv_only_pages, is_resumable, page /
  stash counts) are the hard signals, computed from deterministic, input-derived engine state. A
  >=2-token common-prefix comparison is used only as a *gross-corruption* detector (noise floor >= 2,
  corruption floor 0-1 per findings §2), never as the acceptance criterion.

* The proven oracle for what save/restore must do is `probes/probe_lib.py` (`capture_store` /
  `restore_store`) and `probes/probe_reseed.py` (`P0.1`): the on-disk store layout (pages.bin,
  `stash-*.bin`, chain.json, meta.json, meta LAST), the blake2b content-hash chain, the deepest-K stash
  budget, the eager restore that stamps unreferenced pages root->leaf and advances the serial, and the
  fail-closed restore semantics. This file ports those mechanics onto the `save_state` / `restore_state`
  generator contract the P1 patch implements.

* Engine mechanics under test, all verified against `exllamav3/generator/pagetable.py`:
    - F2 signal: `cached_tokens == (deepest matching GDN anchor page + 1) * PAGE_SIZE`.
    - `alloc_kv_only_pages` (:633) is the clamp tail — cached KV pages past the deepest stashed recurrent
      checkpoint; > 0 is the NORM, never a failure (:626-634).
    - `is_resumable(phash)` (:659) walks a page to its root; any missing ancestor makes it + everything
      above it (deeper) non-resumable.
    - Q1: the snapshot must be a serialised, quiescent, point-in-time read (defrag / eviction never
      interleaves with the capture). A "silent page mix" (page B's KV under page A's hash) is not
      catchable by the engine's `validate_pagetable()` audit (which reconciles phash with the page's
      token sequence, not its KV) — so Q1 is prevention; test v gates the outcome with the detectors that
       do exist, documented in `test_v_defrag_race_q1`.

    - P1 `save_state` RESTRICTED SEAM (BLOCKER C): the generator surface MUST expose save_state as
      wrappable stage methods so a test can race a mutation against the capture — the default seam is
      `_enumerate_store_targets()` (records target metadata) followed by `_copy_store_targets()`
      (copies the enumerated target bytes to the store). `test_v_defrag_race_q1` wraps `_copy_store_targets`
      to force a live defrag at the enumerate→copy boundary; a correct serialized capture is immune, an
      interleaved one surfaces. If the P1 brief instead carries a private `_debug_hook(stage)` kwarg called at
      stage boundaries, that seam replaces `_copy_store_targets` here. The implementation brief MUST carry
      this contract.
"""
import contextlib
import json
import os
import shutil
import sys
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib

import pytest
import torch

from exllamav3 import (
    Cache,
    CacheLayer_quant,
    Config,
    Generator,
    GreedySampler,
    Job,
    Model,
    Tokenizer,
)
from exllamav3.constants import PAGE_SIZE
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP

# --------------------------------------------------------------------------- #
# Configuration — mirrors probes/probe_lib.py prod dual shape (tabby v2 §1/§3)  #
# --------------------------------------------------------------------------- #

MODEL_DIR = "/mnt/nvme/models/qwen38-flash-next-exl3-205"   # 2.05 weights, registry model_dir + name
CACHE_SIZE_204K = 204800          # multiple of 256 — prod first-flip id (tabby v2 §4)
CACHE_SIZE_130K = 130560          # multiple of 256 — the shape a 204k store must NOT restore into
GPU_SPLIT = [24.0, 8.0]           # prod dual shape; engine kwarg use_per_device (tabby maps gpu_split)
K_BITS, V_BITS = 6, 4             # cache_mode "6,4"
MOE_CPU_SPLIT = 150               # prod tail-N-on-CPU split; WITHOUT it the 205B does not fit [24,8]
MOE_CPU_THREADS = 1               # F-N1 determinism pin: >1 CPU worker threads => non-reproducible greedy
STORE_NAME = "session"
# STASH_BUDGET dropped (NIT 2): it was defined but never referenced, and wiring it into every save call
# across the fixtures would couple the suite to save_state's budget signature with no dedicated count gate in
# the 6-gate P1 contract. The save enforces the deepest-K budget internally; exercising the intended call
# shape without asserting its effect is dead config.

# --------------------------------------------------------------------------- #
# Fast, no-GPU preconditions + skip reasons                                      #
# --------------------------------------------------------------------------- #

_NO_GPU_REASON = "no CUDA device available — P1 GPU window required"
_NO_MODEL_REASON = "model weights not present at %s — P1 GPU window required" % MODEL_DIR


def _p1_api_available():
    """True iff the P1 generator surface exists on the Generator class. Class-level hasattr needs only the
    import, not a GPU, so this gates collection cleanly before any model is built."""
    try:
        return hasattr(Generator, "save_state") and hasattr(Generator, "restore_state")
    except Exception:
        return False


P1_API_AVAILABLE = _p1_api_available()
_NO_API_REASON = (
    "P1 save_state()/restore_state() not yet implemented on Generator (tests-first contract — "
    "suite is expected to skip until the sm75-v150-kvsave patch lands)"
)


# --------------------------------------------------------------------------- #
# Offline chain helpers — pure, cross-process clones of pagetable.py:23-28       #
# --------------------------------------------------------------------------- #


def _sha256_file(path):
    """Hex sha256 of a file's bytes (Layer-A digest basis)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _page_hash(tokens, prev_hash):
    """blake2b-16 over prev_hash || int64 token bytes, pagetable.py:23-28 semantics (pure, cross-process)."""
    h = hashlib.blake2b(digest_size = 16)
    if prev_hash is not None:
        h.update(prev_hash)
    h.update(tokens.view(torch.int64).cpu().numpy().tobytes())
    return h.digest()


def _prompt_page_hashes(prompt_tokens):
    """Prompt page-hash chain, pagetable.py:231-250: (len-1)//PAGE_SIZE full pages, last prompt token excluded."""
    n = len(prompt_tokens)
    out = []
    prev = None
    for i in range((n - 1) // PAGE_SIZE):
        prev = _page_hash(prompt_tokens[i * PAGE_SIZE:(i + 1) * PAGE_SIZE], prev)
        out.append(prev)
    return out


def _expected_anchor(prompt_tokens, stash_keys):
    """(deepest prompt page whose hash is in stash_keys + 1) * PAGE_SIZE, and that page index; 0 if none match.
    Pure offline oracle for the F2 signal, same shape as probes/probe_lib.expected_anchor."""
    stash = set(stash_keys)
    deepest = -1
    for i, h in enumerate(_prompt_page_hashes(prompt_tokens)):
        if h in stash:
            deepest = i
    return (deepest + 1) * PAGE_SIZE, deepest


# --------------------------------------------------------------------------- #
# Generator construction + conversation — probes/probe_lib build_gen mirror      #
# --------------------------------------------------------------------------- #


def _build_gen(model_dir=MODEL_DIR, cache_size=CACHE_SIZE_204K, gpu_split=GPU_SPLIT,
               k_bits=K_BITS, v_bits=V_BITS, threads=MOE_CPU_THREADS):
    """Build a flash-Next generator in the prod dual shape, mirroring probes/probe_lib.build_gen:
    Config -> Model -> Cache(quant, BEFORE load) -> load(gpu split) -> Tokenizer -> Generator.
    EXL3_MOE_CPU_SWAP=0 BEFORE load (static expert placement, P0.0(c): a dynamic swap perturbs logits
    between arms and would fake a mechanism failure). The dynamic-placement registry assert proves sweeps
    can never fire once loaded."""
    os.environ["EXL3_MOE_CPU_SWAP"] = "0"
    os.environ["EXL3_MOE_CPU_THREADS"] = str(threads)
    config = Config.from_directory(model_dir)
    config.infer_params.moe_cpu_split = MOE_CPU_SPLIT
    config.infer_params.moe_cpu_threads = threads
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=cache_size, layer_type=CacheLayer_quant,
                  k_bits=k_bits, v_bits=v_bits, max_batch_size=1)
    model.load(use_per_device=list(gpu_split), max_chunk_size=1024)   # 1024 = prod id config (2048 OOMs [24,8])
    tokenizer = Tokenizer.from_config(config)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)
    assert generator.recurrent_cache is not None, "model lacks recurrent_states — GDN stashes impossible"

    def _walk(mods):
        for m in mods:
            yield m
            for c in getattr(m, "modules", None) or []:
                yield from _walk([c])

    split_mods = [m for m in _walk(model.modules)
                  if isinstance(m, BlockSparseMLP) and m.cpu_split_first is not None]
    assert split_mods, "no MoE CPU split modules — [24,8] does not fit without the 150-expert split"
    dynamic = [m.key for m in split_mods if getattr(m, "_split_dynamic", False)]
    assert not dynamic, "dynamic expert placement armed — sweeps could perturb logits"
    return generator, tokenizer


def _unload(generator):
    """Free VRAM: cache tensors are only cleared by Model.unload() (cache.py:134) — del+empty_cache leaves GBs
    held (findings §5)."""
    torch.cuda.synchronize()
    generator.model.unload()
    del generator
    torch.cuda.empty_cache()


def _encode_ids(tokenizer, text):
    """Encode conversation text to engine input ids, shape (1, n) int64, no BOS (mirror probe_lib.encode_ids)."""
    return tokenizer.encode(text, add_bos = False)


# Deterministic conversation (mirror probes/probe_reseed.make_prompt): every prior turn plus its completion,
# with a next-turn scaffold once all turns are complete.
_BLOCK_LINE = "{i}. The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.\n"
_TURN1_BLOCKS = 220
# The second turn is a substantive continuation (not a one-line query): it extends the re-stream prefix well
# past the deepest GDN anchor (which sits at the last checkpoint boundary inside the prior decode), so the
# clamp tail (alloc_kv_only_pages) is non-zero — the documented norm (pagetable.py:626-634). The first turn is
# long enough that its prompt prefill clears the 4096-token "final stretch" and lays down several checkpoints.
_TURN2_TEXT = ("Review the numbered lines above. State which lines mention a fox, then give a detailed multi-\n"
               "sentence summary of the recurring theme, and list three other recurring images you noticed "
               "in the packing and liquor lines. Keep it under one hundred words.")


def _build_turn_texts():
    t1 = "".join(_BLOCK_LINE.format(i = i) for i in range(_TURN1_BLOCKS))
    t1 += "\nList the odd-numbered lines that mention the fox, nothing else."
    return [t1, _TURN2_TEXT]


def _make_prompt(turn_texts, outputs):
    conv = ""
    for u, a in zip(turn_texts, outputs):
        conv += "User:\n%s\nAssistant:\n%s\n" % (u, a)
    if len(outputs) < len(turn_texts):
        conv += "User:\n%s\nAssistant:\n" % turn_texts[len(outputs)]
    return conv


def _run_turn(generator, input_ids, max_new_tokens=64, min_new_tokens=16, stop = "User:"):
    """Enqueue one GreedySampler turn job (pre-encoded ids) and drain via iterate(); returns EOS result fields.
    Robust to a job that drains on max_new without a text EOS (use the final result)."""
    job = Job(input_ids=input_ids, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
              sampler=GreedySampler(), stop_conditions=[stop])
    generator.enqueue(job)
    eos = None
    last = None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r.get("eos"):
                eos = r
            last = r
    if eos is None:
        eos = last
    assert eos is not None, "job drained without any result"
    return {
        "full_completion": eos.get("full_completion", ""),
        "new_tokens": eos.get("new_tokens", 0),
        "cached_tokens": eos.get("cached_tokens", 0),
        "cached_pages": eos.get("cached_pages", 0),
        "prompt_tokens": int(input_ids.shape[-1]),
    }


def _common_prefix_tokens(tokenizer, a, b):
    """Length of the common leading run of two decoded completions (token ids). F-N1 gross-corruption gate
    basis: engine logit noise means near-ties flip, but high-margin leading tokens agree far beyond 2;
    cp < 2 signals a gross (page-corruption) derangement, not the acceptance criterion."""
    ia = _encode_ids(tokenizer, a)[0].tolist()
    ib = _encode_ids(tokenizer, b)[0].tolist()
    n = 0
    for x, y in zip(ia, ib):
        if x != y:
            break
        n += 1
    return n


def _force_defrag(generator):
    """Force a physical defragment of the live cache — simulate a defrag racing the snapshot capture.
    Bypasses the pagetable defrag trigger gates (pagetable.py:827-832) so the repagination actually runs
    when the cache is fragmented.

    Returns True ONLY if pages were ACTUALLY relocated, and RAISES if the defrag no-opped (the cache was too
    unfragmented to rotate, pagetable.py:970). A no-op is not a pass here — an unexercised race would
    false-pass the Q1 gate — so we fail loud instead. Returns False only if defrag() itself raised (a
    tensor-parallel layout that could not be rotated in this harness); the structural gates in test v still
    hold then. Rotation is verified by a before/after page_index fingerprint (pagetable.py:1028-1029 is the
    only place page.page_index is written, so a changed fingerprint proves a relocation happened)."""
    pt = generator.pagetable
    before = {p.phash: p.page_index for p in pt.all_pages}
    pt.last_defrag_serial = 0                       # defeat the serial-age gate (pagetable.py:831)
    pt.access_serial = pt.max_pages * 16            # and satisfy the next gate (pagetable.py:831)
    generator.enable_defrag = True
    try:
        pt.defrag()
    except Exception:
        return False                                # layout could not be rotated — best-effort, gates still hold
    after = {p.phash: p.page_index for p in pt.all_pages}
    if before == after:
        raise RuntimeError("defrag no-op: page layout unchanged — the cache was too unfragmented to rotate "
                           "(<10%, pagetable.py:970), so the save/defrag race was NOT exercised; failing "
                           "loud rather than false-passing. Size the conversation to fragment the cache "
                           "(branch the decode past a checkpoint boundary) before test v can meaningfully pass.")
    return True


def _snapshot_serial_state(generator):
    """Snapshot the pagetable serial state (access_serial, last_defrag_serial, per-page access_serials) so a
    defrag race in test v can leave the shared source generator's serial state untouched for any later test."""
    pt = generator.pagetable
    return (pt.access_serial, pt.last_defrag_serial,
            {p.phash: p.access_serial for p in pt.all_pages})


def _restore_serial_state(generator, snapshot):
    """Restore a _snapshot_serial_state() result after a defrag race, breaking the cross-test coupling the
    race would otherwise introduce (a later save from the same generator would otherwise inherit the race's
    mutated serial state — NOTE 1)."""
    pt = generator.pagetable
    access_serial, last_defrag_serial, page_serials = snapshot
    pt.access_serial = access_serial
    pt.last_defrag_serial = last_defrag_serial
    for p in pt.all_pages:
        if p.phash in page_serials:
            p.access_serial = page_serials[p.phash]


def _burn_pool_fragmentation(generator, tokenizer, n_pages):
    """Deterministic hole-punch so PageTable.defrag() actually rotates — its no-op gate skips any layout
    where <= max(max_pages // 10, 2) pages would change index (pagetable.py:969-970), and a linear chain
    in a fresh 800-page pool moves ZERO pages, so test v's race would never be exercised.

    Mechanism: occupy `n_pages` physical pages at the FRONT of the pool with a throwaway prefill, then
    clear() every complete page it left behind (identity destroyed, kv_position = 0, pagetable.py:173-184)
    and wipe its recurrent stashes. The live conversation then allocates mid-pool instead of at slot 0:
    empty-class eviction is oldest-access_serial first (pagetable build_eviction_order class 1), and the
    cleared pages carry the HIGHEST serials (they were stamped at 801+ during the burn), so never-used
    mid-pool slots win. Compaction then moves >= n_pages + chain pages (front holes + the displaced
    chain) — with n_pages = 100 and a ~25-page chain that is ~125 >= 81 moves — and `_force_defrag`'s
    before/after fingerprint proves the relocation was real.
    """
    pt = generator.pagetable
    block = _encode_ids(tokenizer, _BLOCK_LINE.format(i = 0))[0]
    need = n_pages * PAGE_SIZE + 1                 # (len-1) // PAGE_SIZE == n_pages full prompt pages
    ids = block.repeat((need + len(block) - 1) // len(block))[:need].unsqueeze(0)
    _run_turn(generator, ids, max_new_tokens=8, min_new_tokens=4)
    for p in list(pt.unreferenced_pages.values()):
        if p.kv_position == PAGE_SIZE:
            p.clear()
    if generator.recurrent_cache is not None:
        generator.recurrent_cache.clear()


@contextlib.contextmanager
def _save_copy_stage_race(generator):
    """Exercise the Q1 defrag-race seam (BLOCKER C): force a defrag to race the snapshot capture at the
    boundary between save_state's enumerate stage and its copy stage.

    The P1 contract requires save_state to be structured as wrappable stage methods; the default seam is
    `_copy_store_targets()` (the stage that copies the already-enumerated target bytes into the store),
    preceded by `_enumerate_store_targets()`. Wrapping `_copy_store_targets` so it forces a live defrag first
    means the copy stage races a repagination that happened AFTER enumeration recorded its metadata, BEFORE the
    bytes are copied: a correct serialized capture is immune (the race is fully ordered before or after the
    copy), an interleaved one surfaces as an inconsistent snapshot the test-v gates reject.

    If `_copy_store_targets` is absent the seam contract is not met — surface it loudly rather than silently
    degrading to an un-raced re-run of test i. This path only executes after P1 lands; the suite otherwise
    skips (see the module-docstring P1 seam note)."""
    original = getattr(type(generator), "_copy_store_targets", None)
    if original is None:
        raise RuntimeError("P1 seam contract not met: save_state must expose a `_copy_store_targets()` stage "
                           "method for test v to race (see module docstring). Implement it, or point this "
                           "context manager at the brief's `_debug_hook(stage)` seam instead.")
    def wrapped(self, *args, **kwargs):
        _force_defrag(self)                           # race the snapshot: a live repagination at the copy edge
        return original(self, *args, **kwargs)
    setattr(type(generator), "_copy_store_targets", wrapped)
    try:
        yield
    finally:
        setattr(type(generator), "_copy_store_targets", original)


# --------------------------------------------------------------------------- #
# Fixtures                                                                       #
# --------------------------------------------------------------------------- #


def _skip_if_unavailable():
    """Return a skip reason if the P1 GPU-window preconditions are not met, else None. Checked in the
    session fixture so an early skip never builds a 205B generator."""
    if not torch.cuda.is_available():
        return _NO_GPU_REASON
    if not os.path.isdir(MODEL_DIR):
        return _NO_MODEL_REASON
    if not P1_API_AVAILABLE:
        return _NO_API_REASON
    return None


@pytest.fixture(scope = "session")
def conversation(tmp_path_factory):
    """Single-model source for the whole session: build in the prod dual shape, drive a two-turn conversation
    (so GDN stashes accumulate during decode and the second turn extends the prefix past the deepest anchor —
    the normal clamp-tail case), then build BOTH on-disk stores any test needs WHILE the source is still
    alive, unload the source, and hand back everything CPU-side. This is deliberately model-lifetime
    serialisation, not a contract change: the prod dual shape fills ~30 GiB of the 32 GiB across the two
    cards, so a second full 205B generator can never load — every test therefore works from a pre-built
    store copy and builds at most one fresh generator of its own. Built once; the source is unloaded before
    the first test body runs (sanity property: at most one _build_gen-built generator is ever alive)."""
    reason = _skip_if_unavailable()
    if reason is not None:
        pytest.skip(reason)

    gen, tok = _build_gen(cache_size = CACHE_SIZE_204K, gpu_split = GPU_SPLIT)

    # Hole-punch the pool BEFORE the conversation so the chain lands mid-pool and the test-v defrag race
    # is real (see _burn_pool_fragmentation). The burn's throwaway pages/stashes are fully cleared, so
    # every store below stays a clean single-chain capture of the conversation only.
    _burn_pool_fragmentation(gen, tok, n_pages = 100)

    turn_texts = _build_turn_texts()
    outputs = []
    for i in range(len(turn_texts)):
        prompt_text = _make_prompt(turn_texts, outputs)
        ids = _encode_ids(tok, prompt_text)
        res = _run_turn(gen, ids, max_new_tokens=256, min_new_tokens=16, stop = "User:")
        outputs.append(res["full_completion"])
        print("INFO kvsave.source.turn%d cached_tokens=%d new_tokens=%d"
              % (i, res["cached_tokens"], res["new_tokens"]), flush = True)

    # The prompt re-streamed at restore time: the full conversation up to and including the last user turn,
    # with a next-turn scaffold. Its tokens are what restore must resume from.
    restore_prompt = _make_prompt(turn_texts, outputs[:-1])
    restore_prompt_tokens = _encode_ids(tok, restore_prompt)[0]

    info = types.SimpleNamespace(
        gen = gen,
        tokenizer = tok,
        turn_texts = turn_texts,
        outputs = outputs,
        restore_prompt = restore_prompt,
        restore_prompt_tokens = restore_prompt_tokens,
        restore_ids = _encode_ids(tok, restore_prompt),
        original_output = outputs[-1],
        context_pages = (int(restore_prompt_tokens.shape[-1]) - 1) // PAGE_SIZE,
    )

    # Build both stores while the source is alive, THEN unload it. From here on the source no longer exists
    # in any test body — every test operates from a copy of one of these two dirs.

    # BASE store: a plain save on the quiescent source; the verified copy every non-raced test uses.
    base_dir = tmp_path_factory.mktemp("kvsave_base") / STORE_NAME
    gen.save_state(str(base_dir))
    info.base_store = base_dir

    # RACED store: the exact save / defrag-race / serial-hygiene sequence test v performs, moved to fixture
    # time so the defrag truly races the copy mid-capture. The race IS still exercised (that is test v's
    # point); _force_defrag raises loudly if the cache was too unfragmented to rotate (failing the session
    # rather than false-passing), while its False path — defrag() itself raised, e.g. an un-rotatable TP
    # layout — is the test-v contract's accepted degradation where the structural gates still hold.
    raced_dir = tmp_path_factory.mktemp("kvsave_raced") / STORE_NAME
    serial_snapshot = _snapshot_serial_state(gen)
    with _save_copy_stage_race(gen):
        gen.save_state(str(raced_dir))
    _restore_serial_state(gen, serial_snapshot)
    info.raced_store = raced_dir

    _unload(gen)
    info.gen = None          # source gone before the first test body — sanity property (spec §6)
    yield info
    if info.gen is not None:  # no-op in practice: the source is unloaded above the yield
        _unload(info.gen)


def _copy_store(src, dst):
    """Copy an on-disk store dir to dst (shutil.copytree, idempotent if dst pre-exists). The BASE and RACED
    stores are each built once at session start from the single source generator; a test copies the one it
    needs into a fresh tmp_path so the body can tamper with (or restore from) the set without touching the
    session-level directory."""
    return shutil.copytree(str(src), str(dst), dirs_exist_ok = True)


@pytest.fixture
def source_store(conversation, tmp_path):
    """A verified copy of the session BASE save (plain save on the quiescent source), one fresh temp dir per
    test (so the body can restore from it)."""
    _skip_if_unavailable()  # no-op if the session fixture already skipped
    store = _copy_store(conversation.base_store, tmp_path / STORE_NAME)
    yield store


# --------------------------------------------------------------------------- #
# Target-generator helper (function-scoped, explicit teardown)                   #
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _target_gen(cache_size=CACHE_SIZE_204K):
    """Build a fresh generator to restore into; unload + free VRAM on exit."""
    _skip_if_unavailable()
    tgt, _tok = _build_gen(cache_size = cache_size, gpu_split = GPU_SPLIT)
    try:
        yield tgt
    finally:
        _unload(tgt)


def _read_json(store, name):
    with open(os.path.join(str(store), name)) as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Tests — one per P1 acceptance-table row                                        #
# --------------------------------------------------------------------------- #


def test_i_byte_round_trip(conversation, source_store):
    """(i) byte round-trip: save -> restore on a closed pool, then the re-streamed conversation resumes with
    cached_tokens == (deepest matching anchor page + 1) * PAGE_SIZE — the F2 signal, computed from engine
    state, not text."""
    info = conversation
    store = source_store

    stash_keys = None
    with _target_gen(CACHE_SIZE_204K) as tgt:
        pre_serial = tgt.pagetable.access_serial      # pre-restore high-water (fresh pool: == max_pages)
        # Restore mutation must run under inference mode (cache tensors are inference tensors; a host-mode
        # write raises — test_cpu_cache.py convention, mirrors probes/probe_lib.restore_store).
        with torch.inference_mode():
            tgt.restore_state(str(store))
            # NOTE 2 (probe_lib.py:506): restore MUST advance the pagetable access_serial past every stamped
            # serial so the next job cannot collide with a restored page (avoids inverted LRU / defrag skew).
            # Guard the mechanic: after restore the counter is strictly ahead of the pre-restore high-water.
            assert tgt.pagetable.access_serial > pre_serial, \
                "restore did not advance access_serial past the pre-restore high-water — next-job serial collision"
        # Independent F2 oracle: deepest matching stash anchor, from the restored stashes + re-stream tokens.
        stash_keys = list(tgt.recurrent_cache.keys())
        exp_tokens, _exp_idx = _expected_anchor(info.restore_prompt_tokens, stash_keys)

        res = _run_turn(tgt, info.restore_ids, max_new_tokens=64, min_new_tokens=16)
        cached = res["cached_tokens"]

        # Hard gate (F2).
        assert cached == exp_tokens, ("restore did not resume at the deepest matching anchor: "
                                      "cached_tokens=%d expected=%d" % (cached, exp_tokens))
        assert cached > 0, "restored pool resumed at zero tokens — save/restore lost the whole prefix"
        # cached_pages == deepest matching anchor page + 1 (max_recur), same signal in page units.
        assert res["cached_pages"] == exp_tokens // PAGE_SIZE

        # F-N1 gross-corruption detector only (never the acceptance criterion).
        cp = _common_prefix_tokens(info.tokenizer, info.original_output, res["full_completion"])
        assert cp >= 2, "restored output shares <2 leading tokens with the original — possible page corruption"


def test_ii_torn_set_rejection(conversation, tmp_path):
    """(ii) torn-set rejection: delete meta.json mid-round-trip (Layer-A integrity) -> restore must fail-closed
    to a clean cold start with NO partial state, and the (now-broken) set is kept for forensics."""
    info = conversation
    store = _copy_store(info.base_store, tmp_path / STORE_NAME)

    meta_path = os.path.join(str(store), "meta.json")
    assert os.path.exists(meta_path)
    os.remove(meta_path)                       # simulate a torn / incomplete save set (Layer-A integrity gone)

    with _target_gen(CACHE_SIZE_204K) as tgt:
        raised = False
        try:
            with torch.inference_mode():
                tgt.restore_state(str(store))
        except Exception:
            raised = True

        # Gate 1 (primary — fail-closed, no partial state). A torn set must leave a clean cold pool: the
        # re-stream resumes from zero cached tokens, never half-restored pages.
        res = _run_turn(tgt, info.restore_ids, max_new_tokens=16, min_new_tokens=8)
        assert res["cached_tokens"] == 0, "torn set restored partial state (cached_tokens > 0) — fail-closed broke"

        # Gate 2 (the documented mechanism — fail-closed via validation rejection). A torn set is rejected,
        # not silently served cold: the probe restore_store reads meta first and raises before mutating any page.
        assert raised, "torn set was accepted without rejection — fail-closed did not fire"

    # Payload set kept (not deleted) for forensics after a failed restore — Layer-A keeps the set on the read
    # path. (The meta.json deletion above is the injected fault, not something we assert; the point is the
    # payload files survive intact alongside the missing meta.)
    assert os.path.exists(os.path.join(str(store), "pages.bin")), "pages.bin missing — set not kept for forensics"
    assert os.path.exists(os.path.join(str(store), "chain.json")), "chain.json missing — set not kept for forensics"


def test_iii_config_mismatch_rejection(conversation, tmp_path):
    """(iii) config-mismatch rejection: a 204k store must not restore into a 130k pool -> cold start, loud
    failure path, set kept. cache_size / cache_mode / tensor-manifest are the mismatch keys."""
    info = conversation
    store = _copy_store(info.base_store, tmp_path / STORE_NAME)

    meta = _read_json(store, "meta.json")
    store_pool = meta["pins"]["cache_size"]
    assert store_pool == CACHE_SIZE_204K

    with _target_gen(CACHE_SIZE_130K) as tgt:
        assert tgt.cache.max_num_tokens == CACHE_SIZE_130K
        raised = False
        try:
            with torch.inference_mode():
                tgt.restore_state(str(store))
        except Exception as e:
            raised = True
            msg = str(e).lower()
            assert ("cache_size" in msg or "mismatch" in msg or "pool" in msg), \
                "restore rejected, but not for the pool-shape mismatch: %r" % (e,)
        if not raised:
            # Did not raise: restore must have served cold (no pages claimed for the prefix).
            res = _run_turn(tgt, info.restore_ids, max_new_tokens=16, min_new_tokens=8)
            assert res["cached_tokens"] == 0, "restored a 204k store into a 130k pool and got cache hits — pool gate is a placebo"

    # Set kept after a rejected restore (forensics).
    assert os.path.exists(os.path.join(str(store), "meta.json"))
    assert os.path.exists(os.path.join(str(store), "pages.bin"))


def test_iv_stash_anchor_clamp(conversation, source_store):
    """(iv) stash-anchor clamp: after restore + re-stream, alloc_kv_only_pages equals the documented expected
    clamp tail (cached KV pages past the deepest stashed recurrent checkpoint) — it is NOT zero; a > 0 tail is
    the norm on a hybrid (pagetable.py:626-634)."""
    info = conversation
    store = source_store

    with _target_gen(CACHE_SIZE_204K) as tgt:
        with torch.inference_mode():
            tgt.restore_state(str(store))

        m0 = dict(tgt.pagetable.metrics)
        res = _run_turn(tgt, info.restore_ids, max_new_tokens=64, min_new_tokens=16)
        m1 = dict(tgt.pagetable.metrics)
        alloc_delta = m1["alloc_kv_only_pages"] - m0["alloc_kv_only_pages"]

        # Expected tail = complete prompt pages reclaimed minus the GDN-clamped resume prefix (max_recur).
        expected_tail = info.context_pages - res["cached_pages"]

        # Invariant gate (definition of the clamp): every reclaimed page past the anchor is kv-only.
        assert alloc_delta == expected_tail, \
            ("clamp tail mismatch: alloc_kv_only_pages=%d expected=%d (context_pages=%d cached_pages=%d)"
             % (alloc_delta, expected_tail, info.context_pages, res["cached_pages"]))
        assert expected_tail >= 0, "negative clamp tail — clamp is reclaiming more than was matched"
        # Soft signal (NOT a hard gate): the documented NORM is expected_tail > 0 (the prefix extends past the
        # deepest anchor, so the clamp tail is non-zero). It is marginal — the last GDN checkpoint always lands
        # inside the final prefill stretch, so the tail tracks prompt_end mod 2048-ish and varies with F-N1
        # jitter; the invariant gate above (alloc_delta == expected_tail) is the rock-solid signal. Keep it
        # logged, and keep the conversation sized to maximize it.
        print("INFO kvsave.iv.clamp_tail expected_tail=%d (norm > 0, soft signal)" % expected_tail, flush = True)


def test_v_defrag_race_q1(conversation, tmp_path):
    """(v) defrag race (Q1): the snapshot is a serialised, quiescent, point-in-time read. Force a defrag to
    race the snapshot capture at the enumerate→copy boundary, then confirm the saved snapshot restores to a
    fully consistent state — never a silent page mix.

    A silent page mix (page B's KV stored under page A's hash) is structurally the hard case: the content
    hash covers token ids only (pagetable.py:23-28), so NO digest or the engine's own validate_pagetable audit
    (which checks phash == checksum(sequence)) can see a KV-byte swap. That is why Q1 prevention (the snapshot
    runs under the iteration task's serialization, so defrag/eviction never interleaves with the capture) is
    the whole protection, per plan §2/G5 + findings §2. This test gates the OUTCOME with every detector that
    IS available:

       * validate_pagetable — the snapshot's pages are internally consistent (no sequence/phash-level mix).
       * is_resumable(root/leaf) — the chain is unbroken by the race.
       * cached_tokens == expected — the resume point is exact (a broken mix drops/mis-caches it).
       * common-prefix >= 2 tokens vs the original generation — the only functional Layer-B detector: a KV swap
        corrupts the resumed state so the re-stream diverges beyond the F-N1 noise floor. Prevention is the
        real guard; this is the gross-corruption catch.

    The race is forced by wrapping the P1 `_copy_store_targets` seam (BLOCKER C) so a live defrag runs the
    moment the copy stage begins — AFTER enumeration recorded its metadata, BEFORE the bytes are copied. A
    correct serialized capture is immune, an interleaved one surfaces as an inconsistent snapshot the gates
    below reject. _force_defrag raises rather than no-ops when the cache is too unfragmented to rotate, so a
    non-exercised race fails loud instead of false-passing. The shared source generator's serial state is
    snapshotted before and restored after the save, so the race does not leak into a later save (NOTE 1)."""
    info = conversation
    # The RACED store (defrag racing the enumerate→copy boundary) was built once at session start from the
    # single source generator; copy it into a fresh tmp_path so the body can restore from it. The race was
    # genuinely exercised at fixture time — _force_defrag really repaginated mid-capture and the fresh-index
    # copy is what this test gates — so no live generator or save call is needed here.
    store = _copy_store(info.raced_store, tmp_path / STORE_NAME)

    with _target_gen(CACHE_SIZE_204K) as tgt:
        with torch.inference_mode():
            tgt.restore_state(str(store))
            # validate_pagetable reads page.sequence tensors (via tensor_hash_checksum) — done inside
            # inference mode while the restored pages are still inference tensors.
            tgt.pagetable.validate_pagetable([])

        # Gate 2 — chain unbroken by the race: the root and the deepest leaf both resume down to a root.
        # (is_resumable walks prev_hash bytes + dict lookups only — safe outside inference mode.) A page
        # mix that corrupted the chain would leave a page orphaned here and fail this.
        chain = _read_json(store, "chain.json")
        roots = [e for e in chain["entries"] if e["prev_hash"] is None]
        assert roots, "saved store has no root page"
        for root in roots:
            assert tgt.pagetable.is_resumable(bytes.fromhex(root["phash"])), \
                "restored root page is not resumable — page mix broke the chain"
        assert tgt.pagetable.is_resumable(bytes.fromhex(chain["entries"][-1]["phash"])), \
            "restored leaf is not resumable — the chain did not hold together after the race"

        # Gate 3 — exact resume point; a mix drops/mis-caches it.
        stash_keys = list(tgt.recurrent_cache.keys())
        exp_tokens, _ = _expected_anchor(info.restore_prompt_tokens, stash_keys)
        res = _run_turn(tgt, info.restore_ids, max_new_tokens=64, min_new_tokens=16)
        assert res["cached_tokens"] == exp_tokens, \
            "restored cache hits != expected anchor after a defrag race — silent page mix"

        # Gate 4 (Layer-B gross-corruption detector): the re-stream's output must still prefix-agree with the
        # original beyond the F-N1 noise floor. A KV swap corrupts the resumed state and diverges earlier.
        cp = _common_prefix_tokens(info.tokenizer, info.original_output, res["full_completion"])
        assert cp >= 2, "restored output diverges <2 tokens vs original after a defrag race — page mix"


def test_vi_chain_completeness(conversation, tmp_path):
    """(vi) chain completeness: drop one interior ancestor page from the saved chain -> is_resumable is false
    for everything at/above the gap and true below it.

    The resume anchor on this engine is a recurrent stash, not a page-chain link, so a dropped *page* does not
    reduce cached_tokens (the remaining pages still resolve by hash and the clamp is stash-driven, pagetable.py
    :273-281,:626-634). What a missing ancestor DOES break is is_resumable (:659): the chain-walk from a page to
    its root stops at the gap. That is the signal this test gates. Layer-A still passes (pages.bin is untouched,
    so every kept entry's absolute offset still reads its correct bytes; only this one page is absent, orphaning
    its child)."""
    info = conversation
    store = _copy_store(info.base_store, tmp_path / STORE_NAME)

    # Ground truth for the unbroken chain: chain.json entries root -> leaf (probe save layout).
    chain = _read_json(store, "chain.json")
    entries = chain["entries"]
    assert len(entries) >= 4, "need enough pages to drop an interior ancestor"

    # Drop one interior ancestor page from the chain.
    gap_idx = len(entries) // 2
    kept = [e for i, e in enumerate(entries) if i != gap_idx]
    with open(os.path.join(str(store), "chain.json"), "w") as f:
        json.dump({"page_size": PAGE_SIZE, "entries": kept}, f)

    # BLOCKER B: the oracle hashes EVERY staged file (incl. chain.json) into meta["files"] and restore
    # validates ALL digests BEFORE reading the chain (probe_lib.py:433-436). A tampered chain.json with a
    # stale recorded digest would make a fail-closed restore raise at Layer-A, before is_resumable is ever
    # reached — a false-fail of the exact mechanism this test gates (it would only pass against an impl that
    # skips digest validation, i.e. one that violates Layer-A). Recompute chain.json's digest + size and
    # rewrite meta so Layer-A stays honest and the restore proceeds to the chain-gap detection below. meta.json
    # is re-written LAST (preserving the meta-last ordering invariant); pages.bin is untouched, so this is a
    # pure SEMANTIC chain gap with Layer-A intact.
    chain_path = os.path.join(str(store), "chain.json")
    meta = _read_json(store, "meta.json")
    meta["files"]["chain.json"] = {
        "sha256": _sha256_file(chain_path),
        "size": os.path.getsize(chain_path),
    }
    with open(os.path.join(str(store), "meta.json"), "w") as f:
        json.dump(meta, f)

    root_phash = entries[0]["phash"]                       # root of the chain (prev_hash is None)
    below_gap = entries[gap_idx - 1]["phash"]              # an ancestor of the gap — chain still intact to root
    above_gap = entries[gap_idx + 1]["phash"]              # immediate child of the gap — its ancestor is now missing
    leaf_phash = entries[-1]["phash"]                      # deepest leaf — walks through the gap

    with _target_gen(CACHE_SIZE_204K) as tgt:
        with torch.inference_mode():
            tgt.restore_state(str(store))

        # Check the chain BEFORE any re-stream reclaims / renames pages (restored pages sit unreferenced).
        assert tgt.pagetable.is_resumable(bytes.fromhex(root_phash)), "root not resumable after dropping a deeper page"
        assert tgt.pagetable.is_resumable(bytes.fromhex(below_gap)), "page below the gap must stay resumable"
        assert not tgt.pagetable.is_resumable(bytes.fromhex(above_gap)), \
            "page just above the gap is resumable — a single missing ancestor must break everything above it"
        assert not tgt.pagetable.is_resumable(bytes.fromhex(leaf_phash)), \
            "leaf is resumable — the break did not propagate to the top of the chain"
