# AUDIT: BackgroundCoordinator System - EVA AI

**Date:** 14.04.2026

---

## EXECUTIVE SUMMARY

| Metric | Score | Details |
|--------|-------|---------|
| **Overall** | **6/10** | Core system is functional but with critical gaps |
| EventBus Integration | 7/10 | Dual integration (legacy + new) |
| DeferredCommandSystem | 7/10 | Commands registered |
| Background Jobs | 7/10 | 3 job types implemented |
| Fault Tolerance | 8/10 | Excellent crash-loop protection |
| Detectors | 0/10 | **CRITICAL: Detectors NOT registered** |

---

## 1. ARCHITECTURE

BackgroundCoordinator (autopilot) in eva_ai/core/background_coordinator.py (890 lines):

- _detectors: List[BaseDetector] - opportunity sources
- _job_types: Dict[str, Type[BaseJob]] - registered jobs
- _active_counts: Dict[str, int] - resource counters
- _job_failures: Dict[str, int] - crash-loop counters
- _job_backing_off: Dict[str, bool] - backoff flags
- _timeline: Deque - event history (max 2000)

Main loop runs with adaptive interval (3-30 seconds).

---

## 2. EVENTBUS INTEGRATION

### Legacy EventSystem (lines 628-632):
- system_ready, system_shutdown, user_activity, component_health_change, training_completed

### New EventBus (lines 641-645):
- SYSTEM_READY, SYSTEM_STOP, COMPONENT_STOPPED, LEARNING_COMPLETED, LEARNING_PROGRESS

### Published Events:
- job_scheduled, job_started, job_completed

**Score: 7/10**

---

## 3. DEFERREDCOMMANDSYSTEM INTEGRATION

Commands registered (lines 653-657):
- autopilot_start, autopilot_stop, autopilot_pause, autopilot_resume (HIGH priority)
- autopilot_status (LOW priority)

Pipeline commands (lines 841-854):
- pipeline_generate_deferred, pipeline_skip_model_c (priority=10)

**Score: 7/10**

---

## 4. BACKGROUND JOBS

| Job | Resource | Cooldown | Purpose |
|-----|----------|----------|---------|
| TrainingJob | GPU | 1200s | Train memory graph |
| WebIndexJob | IO | 300s | Web indexing |
| ModuleRecoveryJob | CPU | 600s | Module recovery |

**Score: 7/10**

---

## 5. DETECTORS (CRITICAL ISSUE)

Implemented:
- LearningOpportunityDetector (15s cooldown) -> TrainingJob
- WebDiscoveryDetector (120s cooldown) -> WebIndexJob
- ModuleRecoveryDetector (30s cooldown) -> ModuleRecoveryJob

**CRITICAL BUG in brain_components.py (lines 486-512):**

Jobs ARE registered:
- brain.background.register_job_type(TrainingJob)
- brain.background.register_job_type(WebIndexJob)
- brain.background.register_job_type(ModuleRecoveryJob)

Detectors are NOT registered - missing:
- brain.background.register_detector(LearningOpportunityDetector())
- brain.background.register_detector(WebDiscoveryDetector())
- brain.background.register_detector(ModuleRecoveryDetector())

Impact: _detectors list is empty, autopilot cannot auto-schedule jobs.

**Score: 0/10**

---

## 6. FAULT TOLERANCE

### Concurrency Limits:
- CPU: 2, GPU: 1, IO: 4

### Cooldowns:
- TrainingJob: 1200s, WebIndexJob: 300s, ModuleRecoveryJob: 600s

### Crash-Loop Protection:
Exponential backoff: min(3600s, 30s * 2^failures)

| Failures | Backoff |
|----------|---------|
| 1 | 60s |
| 2 | 120s |
| 3 | 240s |
| 4 | 480s |
| 5 | 960s |
| 6 | 1920s |
| 7+ | 3600s |

### Resource Thresholds:
- Hard: CPU=0.90, RAM=0.95 (block)
- Soft: CPU=0.80, RAM=0.90 (throttle)

### Idle Detection:
- 10 seconds idle required

**Score: 8/10**

---

## 7. ISSUES

### CRITICAL
1. Detectors not registered - autopilot non-functional
2. Duplicate _do_probe in learning_detector.py (lines 13-56, 58-101)

### HIGH
3. No job success verification
4. Detectors not configurable
5. Timeline not persisted

### MEDIUM
6. Duplicate event handling (legacy + new EventBus)
7. RAM thresholds not in config
8. No health status publishing

---

## 8. FINAL SCORE

| Component | Score | Weight |
|-----------|-------|--------|
| Architecture | 8/10 | 20% |
| EventBus Integration | 7/10 | 15% |
| DeferredCommandSystem | 7/10 | 15% |
| Background Jobs | 7/10 | 15% |
| Detectors | 0/10 | 15% |
| Fault Tolerance | 8/10 | 20% |
| **TOTAL** | | **6/10** |

---

## 9. RECOMMENDATION

Fix detector registration in brain_components.py after line 507:

`
from .opportunities.learning_detector import LearningOpportunityDetector
from .opportunities.web_discovery_detector import WebDiscoveryDetector
from .opportunities.recovery_detector import ModuleRecoveryDetector

brain.background.register_detector(LearningOpportunityDetector())
brain.background.register_detector(WebDiscoveryDetector())
brain.background.register_detector(ModuleRecoveryDetector())
`
