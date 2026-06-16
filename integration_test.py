"""Quick integration test script."""
import sys
sys.path.insert(0, r"C:\Users\TUSHAR\OneDrive\Desktop\vlm_cam")

from main import ARGUS

a = ARGUS()
print("OK: ARGUS initialized")

a.graph.add_person_node(1, "walking person")
a.graph.parse_caption_for_graph(1, "person holding a coffee cup while walking")
print(f"OK: Graph stats: {a.graph.get_stats()}")

a.vector_store.store(1, [0.1] * 384, "test")
print(f"OK: Vector count: {a.vector_store.count}")

a.sqlite.log_event("detection", 1, {"persons": 1})
print(f"OK: Events: {len(a.sqlite.get_recent_events())}")

alert = a.alerts.evaluate_new_person(1, "walking person")
print(f'OK: Alert type: {alert["type"] if alert else "none"}')

vss = a.vss
vss.load()
vss.store(1, "person in blue shirt walking")
matches = vss.search_similar(1, "person in blue shirt walking")
print(f"OK: VSS loaded, matches: {len(matches)}")

a.stop()
a.stop()
print("OK: Double-stop idempotent")
print("ALL CHECKS PASSED")
