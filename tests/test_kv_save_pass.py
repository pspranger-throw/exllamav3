"""P2 save-pass / save-policy tests — offline, no GPU, no model.

Gate the two P2 engine-surface additions on `sm75-v150-kvsave`:

* `AsyncGenerator` exposes save/restore as EXPLICIT delegate methods (user decision 2026-09-24:
  no `__getattr__` forwarding), and `request_save()` executes the snapshot inside the iteration
  task (Q1: save and `iterate()` strictly alternate — never interleave).
* `Generator.save_state` skips zero-stash captures (user decision 2026-09-24: "no dead sets" —
  a pages-only store can never restore on a hybrid model) and honors the `stash_budget_mb` knob.

These run without CUDA; the model-bearing acceptance remains the P1 suite
(`test_kv_save_restore.py`, GPU window) and the pre-signed P3 rig drills.
"""

import asyncio
import sys
import os
import types

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest

from exllamav3.generator.generator import Generator
from exllamav3.generator.async_generator import AsyncGenerator


class StubSyncGenerator:
    """Stands in for the sync Generator in AsyncGenerator loop tests."""

    def __init__(self, save_result = None, save_error = None):
        self.save_calls = []
        self.iterate_calls = 0
        self._save_result = save_result if save_result is not None else {"n_pages": 1, "n_stashes": 1}
        self._save_error = save_error

    def save_state(self, store, stash_budget_mb = None):
        self.save_calls.append((store, stash_budget_mb))
        if self._save_error is not None:
            raise self._save_error
        return self._save_result

    def restore_state(self, store):
        return {"restored_from": store}

    def iterate(self):
        self.iterate_calls += 1
        return []


def make_async_gen(stub):
    """AsyncGenerator wired around a stub sync generator, without constructing a real Generator."""
    ag = AsyncGenerator.__new__(AsyncGenerator)
    ag.generator = stub
    ag.jobs = {}
    ag.error = None
    ag.condition = asyncio.Condition()
    ag._save_request = None
    ag.iteration_task = asyncio.create_task(ag._run_iteration())
    return ag


class ExplicitDelegateTests(unittest.TestCase):
    def test_save_and_restore_are_real_methods_not_getattr(self):
        for name in ("save_state", "restore_state", "request_save"):
            self.assertIn(name, vars(AsyncGenerator),
                          f"{name} must be an explicit AsyncGenerator method (no __getattr__ forwarding)")
        self.assertFalse(hasattr(AsyncGenerator, "__getattr__"),
                         "AsyncGenerator must not forward attributes via __getattr__")

    def test_delegates_reach_the_inner_generator(self):
        stub = StubSyncGenerator()

        async def run():
            ag = make_async_gen(stub)
            out = await ag.request_save("/tmp/x", stash_budget_mb = 64)
            await ag.close()
            return out

        result = asyncio.run(run())
        self.assertEqual(result, stub._save_result)
        self.assertEqual(stub.save_calls, [("/tmp/x", 64)])

        async def run_restore():
            ag2 = make_async_gen(StubSyncGenerator())
            got = ag2.restore_state("/tmp/y")
            await ag2.close()
            return got

        self.assertEqual(asyncio.run(run_restore()), {"restored_from": "/tmp/y"})


class SavePassQ1Tests(unittest.TestCase):
    def test_save_runs_in_iteration_task_not_inline(self):
        """request_save must resolve via the loop's save pass, and the save must not interleave
        with iterate(): no iterate() call may happen inside the save window (Q1)."""
        stub = StubSyncGenerator(save_result = {"ok": True})

        async def run():
            ag = make_async_gen(stub)
            fut = ag.request_save("/tmp/s")
            # Nothing may run until the iteration task gets scheduled
            self.assertEqual(stub.save_calls, [])
            out = await fut
            await ag.close()
            return out

        self.assertEqual(asyncio.run(run()), {"ok": True})
        self.assertEqual(stub.save_calls, [("/tmp/s", None)])
        # The save pass consumes the whole pass: nothing to assert about iterate() having run,
        # but it must never have run DURING the save (stub records call order separately below).
        self.assertEqual(stub.iterate_calls, 0)

    def test_save_pass_consumes_its_pass_whole(self):
        """A job enqueued after the save resolves runs on a LATER pass: the save pass never falls
        through into iterate(), so save and iterate strictly alternate (Q1)."""
        order = []

        class OrderStub(StubSyncGenerator):
            def save_state(self, store, stash_budget_mb = None):
                order.append("save")
                return {"ok": True}

            def iterate(self):
                order.append("iterate")
                return super().iterate()

            def enqueue(self, job):
                pass

        stub = OrderStub()

        async def run():
            ag = make_async_gen(stub)
            fut = ag.request_save("/tmp/s")
            await fut
            # A job arriving after the save must land on a later loop pass
            fake_job = object()
            fake_async_job = types.SimpleNamespace(put_result = lambda result: None)
            ag.jobs[fake_job] = fake_async_job
            await ag._notify_condition()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            await ag.close()

        asyncio.run(run())
        self.assertEqual(order[:2], ["save", "iterate"],
                         "the save pass must consume its pass whole: iterate() runs on a LATER pass")

    def test_double_pending_save_is_rejected(self):
        stub = StubSyncGenerator()

        async def run():
            ag = make_async_gen(stub)
            fut = ag.request_save("/tmp/s1")
            with self.assertRaises(RuntimeError):
                ag.request_save("/tmp/s2")
            await fut
            await ag.close()

        asyncio.run(run())

    def test_save_exception_fails_the_future(self):
        stub = StubSyncGenerator(save_error = RuntimeError("boom"))

        async def run():
            ag = make_async_gen(stub)
            fut = ag.request_save("/tmp/s")
            with self.assertRaises(RuntimeError):
                await fut
            await ag.close()

        asyncio.run(run())

    def test_latched_error_rejects_save(self):
        stub = StubSyncGenerator()

        async def run():
            ag = make_async_gen(stub)
            ag.error = RuntimeError("latched")
            with self.assertRaises(RuntimeError):
                ag.request_save("/tmp/s")
            await ag.close()

        asyncio.run(run())


class ZeroStashSkipPolicyTests(unittest.TestCase):
    """Generator.save_state policy: zero-stash captures are skipped (no dead sets)."""

    def _stub_self(self, stashes, targets = 2):
        class Stub:
            pending_jobs = {}
            active_jobs = {}

            def __init__(self):
                self.copied = []

            def _enumerate_store_targets(self):
                return {
                    "targets": list(range(targets)),
                    "gaps": [],
                    "stashes": stashes,
                    "tensors": [],
                    "devices": [],
                    "deepest_anchor_key": None,
                    "deepest_anchor_page_idx": -1,
                }

            def _copy_store_targets(self, store):
                self.copied.append(store)
                return {"dir": store, "n_pages": targets, "n_stashes": len(stashes)}

        return Stub()

    def test_zero_stash_skips_write_and_is_loud(self):
        stub = self._stub_self(stashes = [])
        result = Generator.save_state(stub, "/tmp/kv/store")
        self.assertEqual(result["skipped"], "zero-stash")
        self.assertEqual(result["n_stashes"], 0)
        self.assertIsNone(result["dir"])
        self.assertEqual(stub.copied, [], "a zero-stash save must not write anything")

    def test_nonzero_stash_writes_normally(self):
        stub = self._stub_self(stashes = [("k", {})])
        result = Generator.save_state(stub, "/tmp/kv/store")
        self.assertNotIn("skipped", result)
        self.assertEqual(stub.copied, ["/tmp/kv/store"])

    def test_stash_budget_mb_is_honored(self):
        stub = self._stub_self(stashes = [("k", {})])
        Generator.save_state(stub, "/tmp/kv/store", stash_budget_mb = 64)
        self.assertEqual(stub._stash_budget_bytes, 64 * 1024 ** 2)

    def test_default_budget_is_unset(self):
        stub = self._stub_self(stashes = [("k", {})])
        Generator.save_state(stub, "/tmp/kv/store")
        self.assertIsNone(stub._stash_budget_bytes)

    def test_non_quiescent_save_fails_loud(self):
        stub = self._stub_self(stashes = [("k", {})])
        stub.active_jobs = {"req": object()}
        with self.assertRaises(AssertionError):
            Generator.save_state(stub, "/tmp/kv/store")


if __name__ == "__main__":
    unittest.main()
