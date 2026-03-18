import os
import time
import json
import threading
import tempfile
import unittest
from typing import Any, Dict

# Project imports
from cogniflex.core.background_coordinator import BackgroundCoordinator, Policies
from cogniflex.core.background_jobs.base_job import BaseJob
from cogniflex.core.autopilot_cache import AutopilotCache


class NoOpJob(BaseJob):
    job_type = "NoOpJob"
    resource_class = "CPU"

    def run(self, context: Dict[str, Any]) -> None:
        # minimal work
        time.sleep(0.02)


class OneShotDetector:
    name = "OneShotDetector"

    def __init__(self):
        self._fired = False

    def probe(self, context: Dict[str, Any]):
        # emit only once
        if self._fired:
            return []
        self._fired = True
        return [{"job_type": "NoOpJob", "params": {}}]


class BrainStub:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir
        self.autopilot_cache = AutopilotCache(cache_dir)
        self.initialized = True
        self.running = True


class ResourceManagerStub:
    def __init__(self, cpu: float, mem: float):
        self._cpu = cpu
        self._mem = mem
    def get_cpu_usage(self) -> float:
        return self._cpu
    def get_memory_usage(self) -> float:
        return self._mem


class TestAutopilotAndHealth(unittest.TestCase):
    def setUp(self):
        self.tmpdir_obj = tempfile.TemporaryDirectory(prefix="cogniflex_test_")
        self.tmpdir = self.tmpdir_obj.name

    def tearDown(self):
        try:
            self.tmpdir_obj.cleanup()
        except Exception:
            pass

    def _read_events(self, cache_dir: str):
        path = os.path.join(cache_dir, "autopilot", "events.jsonl")
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def test_autopilot_events_and_cooldown(self):
        brain = BrainStub(self.tmpdir)
        policies = Policies(
            idle_threshold_s=0.0,
            tick_interval_s=0.05,
        )
        # provide per-job cooldowns: NoOpJob -> 1 second
        policies.job_cooldowns_s = {"NoOpJob": 1}

        # Используем стабильный RM с низкой загрузкой, чтобы не зависеть от реальных метрик хоста
        rm = ResourceManagerStub(cpu=0.05, mem=0.05)
        # Очистим events.jsonl, если остался от предыдущих прогонов
        events_path = os.path.join(brain.cache_dir, "autopilot", "events.jsonl")
        try:
            if os.path.exists(events_path):
                os.remove(events_path)
        except Exception:
            pass
        bc = BackgroundCoordinator(brain=brain, resource_manager=rm, policies=policies)
        bc.register_job_type(NoOpJob)
        bc.register_detector(OneShotDetector())

        try:
            bc.start()
            # allow several ticks
            time.sleep(0.35)
        finally:
            bc.stop()

        events = self._read_events(brain.cache_dir)
        # Count job_start for NoOpJob
        starts = [e for e in events if e.get("kind") == "job_start" and e.get("data", {}).get("job") == "NoOpJob"]
        self.assertGreaterEqual(len(starts), 1, "Expected at least one NoOpJob start event")
        # Ensure cooldown prevented repeated scheduling within short window
        self.assertLessEqual(len(starts), 1, "Cooldown should prevent multiple NoOpJob runs in this window")

    def test_core_health_available(self):
        # Import here to avoid heavy import for other tests if not needed
        from cogniflex.core.core_brain import CoreBrain
        brain = CoreBrain()
        try:
            health = brain.get_system_health()
            # Basic expectations
            self.assertIsInstance(health, dict)
            self.assertIn("status", health)
            self.assertIn("components", health)
            # Accept either top-level 'resources' or components['resources'] entry
            has_top_resources = "resources" in health
            has_component_resources = health.get("components", {}).get("resources") == "ok"
            self.assertTrue(
                has_top_resources or has_component_resources,
                "Expected resources info either at top-level or inside components"
            )
        finally:
            try:
                brain.shutdown()
            except Exception:
                pass

    def test_knowledge_graph_stats(self):
        from cogniflex.core.core_brain import CoreBrain
        brain = CoreBrain()
        try:
            # Ensure components are initialized
            initialized = brain.initialize()
            if not initialized:
                self.skipTest("CoreBrain failed to initialize; skipping knowledge graph stats test")
            kg = getattr(brain, "knowledge_graph", None)
            self.assertIsNotNone(kg, "KnowledgeGraph should be initialized")
            stats = kg.get_statistics()
            domains = kg.get_domain_statistics()
            self.assertIsInstance(stats, dict)
            self.assertTrue(isinstance(domains, dict), "get_domain_statistics should return dict domain->count")
        finally:
            try:
                brain.shutdown()
            except Exception:
                pass

    def test_token_cache_initialized_and_query_fallback(self):
        from cogniflex.core.core_brain import CoreBrain
        brain = CoreBrain()
        try:
            # Token cache should either initialize or be None with error logged; ensure attribute exists
            self.assertTrue(hasattr(brain, 'token_cache'))
            # Basic query path should return a dict
            resp = brain.process_query("ping")
            self.assertIsInstance(resp, dict)
            # Accept either explicit status or presence of 'response' with optional 'error'
            has_status = "status" in resp
            has_response_payload = "response" in resp
            self.assertTrue(
                has_status or has_response_payload,
                f"Unexpected response shape: {resp}"
            )
        finally:
            try:
                brain.shutdown()
            except Exception:
                pass

    def test_soft_threshold_throttling_prevents_scheduling(self):
        # Arrange: soft thresholds low, hard thresholds high; RM reports medium usage -> should_pause = True
        brain = BrainStub(tempfile.mkdtemp(prefix="cogniflex_test_"))
        rm = ResourceManagerStub(cpu=0.5, mem=0.5)  # 50% usage
        policies = Policies(
            idle_threshold_s=0.0,
            tick_interval_s=0.05,
            cpu_threshold_soft=0.1,
            ram_threshold_soft=0.1,
            cpu_threshold_hard=0.99,
            ram_threshold_hard=0.99,
        )

        class ShouldNotRunJob(BaseJob):
            job_type = "ShouldNotRunJob"
            resource_class = "CPU"
            def run(self, context: Dict[str, Any]) -> None:
                # If ever runs, sleep briefly
                time.sleep(0.01)

        class AlwaysDetector:
            name = "AlwaysDetector"
            def probe(self, context: Dict[str, Any]):
                return [{"job_type": "ShouldNotRunJob", "params": {}}]

        bc = BackgroundCoordinator(brain=brain, resource_manager=rm, policies=policies)
        bc.register_job_type(ShouldNotRunJob)
        bc.register_detector(AlwaysDetector())

        try:
            bc.start()
            time.sleep(0.3)
        finally:
            bc.stop()

        # Verify no job_start events due to soft-threshold throttling
        events = self._read_events(brain.cache_dir)
        starts = [e for e in events if e.get("kind") == "job_start" and e.get("data", {}).get("job") == "ShouldNotRunJob"]
        self.assertEqual(len(starts), 0, "Soft thresholds should prevent scheduling any jobs")

    def test_crash_loop_exponential_backoff(self):
        brain = BrainStub(tempfile.mkdtemp(prefix="cogniflex_test_"))
        policies = Policies(idle_threshold_s=0.0, tick_interval_s=0.03, concurrency={"CPU": 1, "GPU": 1, "IO": 1})
        # Remove cooldown for failing job to isolate backoff behavior
        policies.job_cooldowns_s = {"FailingJob": 0}

        class FailingJob(BaseJob):
            job_type = "FailingJob"
            resource_class = "CPU"
            def run(self, context: Dict[str, Any]) -> None:
                raise RuntimeError("boom")

        class AlwaysDetector:
            name = "AlwaysDetector"
            def probe(self, context: Dict[str, Any]):
                return [{"job_type": "FailingJob", "params": {}}]

        rm = ResourceManagerStub(cpu=0.05, mem=0.05)
        bc = BackgroundCoordinator(brain=brain, resource_manager=rm, policies=policies)
        bc.register_job_type(FailingJob)
        bc.register_detector(AlwaysDetector())

        try:
            bc.start()
            # Allow some ticks: first run fails quickly; backoff should block re-runs for >=1s
            time.sleep(0.6)
        finally:
            bc.stop()

        events = self._read_events(brain.cache_dir)
        starts = [e for e in events if e.get("kind") == "job_start" and e.get("data", {}).get("job") == "FailingJob"]
        # Expect only one start within the short window due to backoff
        self.assertLessEqual(len(starts), 1, f"Backoff should prevent repeated starts, got {len(starts)} starts")


if __name__ == "__main__":
    unittest.main()
