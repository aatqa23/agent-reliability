"""Runs every offline check one after another (works the same on Windows, Mac and Linux; no API calls).
From the project root, with the venv active:   python check_all.py"""
import subprocess
import sys

CHECKS = ["contract.py", "solve_tasks.py", "solve_tasks_c.py", "test_tools.py", "test_react.py", "test_plan_execute.py",
          "test_llm.py", "test_run.py", "test_faults.py", "test_cache.py", "paraphraser.py --check", "chains.py --check",
          "test_metrics.py", "test_task_classifier.py"]
failed = []
for c in CHECKS:
    r = subprocess.run([sys.executable, *c.split()], capture_output=True, text=True)
    last = (r.stdout.strip().splitlines() or r.stderr.strip().splitlines() or [""])[-1]
    print(f"[{'ok' if r.returncode == 0 else 'FAIL'}] {c}: {last[:90]}")
    if r.returncode:
        failed.append(c)
        print((r.stdout + r.stderr)[-800:])
print("\nALL CHECKS PASSED" if not failed else f"\nFAILED: {failed}")
sys.exit(1 if failed else 0)
