"""
Learning Modules Test Script
Tests each new module independently.
"""
import sys
import os

# Results storage
results = {
    "curiosity_engine": {"works": [], "fails": [], "needs_fixing": []},
    "knowledge_awareness": {"works": [], "fails": [], "needs_fixing": []},
    "online_knowledge": {"works": [], "fails": [], "needs_fixing": []},
    "self_dialog_learning": {"works": [], "fails": [], "needs_fixing": []}
}

# ============================================
# TEST 1: CuriosityEngine
# ============================================
print("=" * 50)
print("TEST 1: CuriosityEngine")
print("=" * 50)

try:
    from eva.learning.curiosity_engine import CuriosityEngine, CuriosityType
    
    engine = CuriosityEngine(brain=None)
    
    # Test 1: detect_curiosity_triggers
    text = "Что такое квантовая механика и как она связана с философией?"
    triggers = engine.detect_curiosity_triggers(text)
    print(f"[OK] Triggers found: {len(triggers)}")
    results["curiosity_engine"]["works"].append("detect_curiosity_triggers")
    for t in triggers:
        print(f"  - {t.trigger_type.value}: {t.topic}")
    
    # Test 2: assess_knowledge_gap
    gap = engine.assess_knowledge_gap("квантовая механика")
    print(f"[OK] Knowledge gap: {gap}")
    results["curiosity_engine"]["works"].append("assess_knowledge_gap")
    
    # Test 3: generate_questions
    questions = engine._generate_questions(CuriosityType.ENTITY_EXPLORATION, "нейронные сети", [])
    print(f"[OK] Generated questions: {questions}")
    results["curiosity_engine"]["works"].append("_generate_questions")
    
    # Test 4: get_curiosity_report
    report = engine.get_curiosity_report()
    print(f"[OK] Curiosity report: {report}")
    results["curiosity_engine"]["works"].append("get_curiosity_report")
    
except Exception as e:
    print(f"[FAIL] CuriosityEngine: {e}")
    results["curiosity_engine"]["fails"].append(str(e))

# ============================================
# TEST 2: KnowledgeAwareness
# ============================================
print("\n" + "=" * 50)
print("TEST 2: KnowledgeAwareness")
print("=" * 50)

try:
    from eva.learning.knowledge_awareness import KnowledgeAwareness
    
    aware = KnowledgeAwareness(brain=None)
    
    # Test marking
    aware.mark_verified("Земля круглая", "Wikipedia")
    aware.mark_generated("Я думаю, значит существую", 0.6)
    print("[OK] mark_verified and mark_generated work")
    results["knowledge_awareness"]["works"].append("mark_verified")
    results["knowledge_awareness"]["works"].append("mark_generated")
    
    # Test status
    status1 = aware.get_status("Земля круглая")
    status2 = aware.get_status("Я думаю, значит существую")
    status3 = aware.get_status("Неизвестный факт")
    print(f"[OK] Status 1 (verified): {status1}")
    print(f"[OK] Status 2 (generated): {status2}")
    print(f"[OK] Status 3 (unknown): {status3}")
    results["knowledge_awareness"]["works"].append("get_status")
    
    # Test get_source_type
    source = aware.get_source_type("Земля круглая")
    print(f"[OK] Source type: {source}")
    results["knowledge_awareness"]["works"].append("get_source_type")
    
    # Test report
    report = aware.get_knowledge_report()
    print(f"[OK] Report: {report}")
    results["knowledge_awareness"]["works"].append("get_knowledge_report")
    
except Exception as e:
    print(f"[FAIL] KnowledgeAwareness: {e}")
    results["knowledge_awareness"]["fails"].append(str(e))

# ============================================
# TEST 3: OnlineKnowledgeAccess
# ============================================
print("\n" + "=" * 50)
print("TEST 3: OnlineKnowledgeAccess")
print("=" * 50)

try:
    from eva.knowledge.online_knowledge import OnlineKnowledgeAccess
    
    online = OnlineKnowledgeAccess(brain=None)
    
    # Test Wikipedia search
    print("[INFO] Testing Wikipedia search (may take a moment)...")
    result = online.search_wikipedia("Python (programming language)", lang='en')
    if result:
        print(f"[OK] Wikipedia result: {result.get('title')}")
        print(f"[OK] Summary: {result.get('summary')[:100]}...")
        results["online_knowledge"]["works"].append("search_wikipedia")
    else:
        print("[WARN] Wikipedia search returned no result (possible network/API issue)")
        results["online_knowledge"]["needs_fixing"].append("search_wikipedia may need better error handling or offline fallback")
    
    # Test fact verification
    verification = online.verify_fact("Python is a programming language")
    print(f"[OK] Verification: verified={verification.verified}, confidence={verification.confidence}")
    results["online_knowledge"]["works"].append("verify_fact")
    
    # Test learn about entity
    learn = online.learn_about_entity("Artificial intelligence")
    if learn:
        print(f"[OK] Learned entity: {learn.get('entity')}")
        print(f"[OK] Verified: {learn.get('verified')}")
        results["online_knowledge"]["works"].append("learn_about_entity")
    else:
        print("[WARN] learn_about_entity returned empty")
        results["online_knowledge"]["needs_fixing"].append("learn_about_entity needs better fallback")
    
except Exception as e:
    print(f"[FAIL] OnlineKnowledgeAccess: {e}")
    results["online_knowledge"]["fails"].append(str(e))

# ============================================
# TEST 4: SelfDialogLearningSystem
# ============================================
print("\n" + "=" * 50)
print("TEST 4: SelfDialogLearningSystem")
print("=" * 50)

try:
    from eva.learning.self_dialog_learning import SelfDialogLearningSystem
    
    # Create without brain (simplified test)
    sdsl = SelfDialogLearningSystem(brain=None, config={'enabled': False})
    
    # Test cycle (without model)
    try:
        cycle = sdsl.run_self_dialog_cycle("тестирование")
        print(f"[OK] Cycle completed: {cycle.cycle_id}")
        print(f"[OK] Quality score: {cycle.quality_score}")
        print(f"[OK] Ethics passed: {cycle.ethics_passed}")
        results["self_dialog_learning"]["works"].append("run_self_dialog_cycle (without brain)")
        
        # Test stats
        stats = sdsl.get_stats()
        print(f"[OK] Stats: {stats}")
        results["self_dialog_learning"]["works"].append("get_stats")
        
        # Test dialog history
        history = sdsl.get_dialog_history()
        print(f"[OK] Dialog history length: {len(history)}")
        results["self_dialog_learning"]["works"].append("get_dialog_history")
        
    except Exception as e:
        print(f"[FAIL] Cycle error: {e}")
        results["self_dialog_learning"]["fails"].append(f"run_self_dialog_cycle: {str(e)}")
    
except Exception as e:
    print(f"[FAIL] SelfDialogLearningSystem init: {e}")
    results["self_dialog_learning"]["fails"].append(str(e))

# ============================================
# Print Summary
# ============================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)

for module, data in results.items():
    print(f"\n### {module.upper()} ###")
    print(f"  Works: {data['works']}")
    if data['fails']:
        print(f"  Fails: {data['fails']}")
    if data['needs_fixing']:
        print(f"  Needs Fixing: {data['needs_fixing']}")
