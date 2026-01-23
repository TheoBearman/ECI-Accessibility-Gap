"""
Exploration script to discover available benchmarks from Epoch AI's Airtable database.
This helps identify which benchmarks have sufficient data for gap analysis.
"""
import os
from collections import defaultdict

# Environment variables should be set before running:
# export AIRTABLE_BASE_ID="appKfIQ7h4TtoGmQw"
# export AIRTABLE_PERSONAL_ACCESS_TOKEN="your_token_here"

if not os.environ.get("AIRTABLE_BASE_ID") or not os.environ.get("AIRTABLE_PERSONAL_ACCESS_TOKEN"):
    print("Error: Please set AIRTABLE_BASE_ID and AIRTABLE_PERSONAL_ACCESS_TOKEN environment variables")
    print("Example:")
    print('  export AIRTABLE_BASE_ID="appKfIQ7h4TtoGmQw"')
    print('  export AIRTABLE_PERSONAL_ACCESS_TOKEN="your_token_here"')
    exit(1)

from epochai.airtable.models import MLModel, Task, Score, Organization, BenchmarkRun, MLModelGroup

def explore_data():
    print("=" * 60)
    print("EPOCH AI BENCHMARK DATA EXPLORATION")
    print("=" * 60)

    # Load all data with memoization
    print("\n[1/6] Loading Tasks (Benchmarks)...")
    tasks = Task.all(memoize=True)
    print(f"  Found {len(tasks)} tasks/benchmarks")

    print("\n[2/6] Loading Benchmark Runs...")
    runs = BenchmarkRun.all(memoize=True)
    print(f"  Found {len(runs)} benchmark runs")

    print("\n[3/6] Loading Scores...")
    scores = Score.all(memoize=True)
    print(f"  Found {len(scores)} scores")

    print("\n[4/6] Loading Models...")
    models = MLModel.all(memoize=True)
    print(f"  Found {len(models)} models")

    print("\n[5/6] Loading Model Groups...")
    model_groups = MLModelGroup.all(memoize=True)
    print(f"  Found {len(model_groups)} model groups")

    print("\n[6/6] Loading Organizations...")
    organizations = Organization.all(memoize=True)
    print(f"  Found {len(organizations)} organizations")

    # Create lookup dictionaries
    task_by_id = {t.id: t for t in tasks}
    model_by_id = {m.id: m for m in models}
    run_by_id = {r.id: r for r in runs}
    model_group_by_id = {mg.id: mg for mg in model_groups}

    # List all tasks/benchmarks
    print("\n" + "=" * 60)
    print("AVAILABLE BENCHMARKS (TASKS)")
    print("=" * 60)

    for task in tasks:
        num_runs = len(task.benchmark_runs) if task.benchmark_runs else 0
        print(f"\n  Task: {task.name}")
        print(f"    Path: {task.path}")
        print(f"    Benchmark runs: {num_runs}")

    # Analyze benchmark runs by task
    print("\n" + "=" * 60)
    print("BENCHMARK RUNS BY TASK")
    print("=" * 60)

    task_model_counts = defaultdict(set)

    for run in runs:
        if run.task:
            # Get the task(s) for this run
            task_list = run.task if isinstance(run.task, list) else [run.task]
            for task_item in task_list:
                task_id = task_item.id if hasattr(task_item, 'id') else task_item
                if run.model:
                    model_list = run.model if isinstance(run.model, list) else [run.model]
                    for model_item in model_list:
                        model_id = model_item.id if hasattr(model_item, 'id') else model_item
                        task_model_counts[task_id].add(model_id)

    print("\nModels evaluated per benchmark:")
    benchmark_info = []
    for task_id, model_ids in sorted(task_model_counts.items(), key=lambda x: len(x[1]), reverse=True):
        task = task_by_id.get(task_id)
        task_name = task.name if task else task_id
        task_path = task.path if task else "unknown"
        print(f"  {task_name}: {len(model_ids)} models")
        benchmark_info.append({
            'id': task_id,
            'name': task_name,
            'path': task_path,
            'num_models': len(model_ids)
        })

    # Analyze model accessibility from model groups
    print("\n" + "=" * 60)
    print("MODEL ACCESSIBILITY ANALYSIS")
    print("=" * 60)

    open_count = 0
    closed_count = 0
    unknown_count = 0

    accessibility_values = set()

    for mg in model_groups:
        accessibility = getattr(mg, 'accessibility', None)
        if accessibility:
            accessibility_values.add(accessibility)
            acc_lower = str(accessibility).lower()
            if 'open' in acc_lower:
                open_count += 1
            elif 'closed' in acc_lower or 'api' in acc_lower:
                closed_count += 1
            else:
                unknown_count += 1
        else:
            unknown_count += 1

    print(f"\n  Open model groups: {open_count}")
    print(f"  Closed model groups: {closed_count}")
    print(f"  Unknown accessibility: {unknown_count}")
    print(f"\n  Unique accessibility values: {accessibility_values}")

    # Sample a model to understand the full data chain
    print("\n" + "=" * 60)
    print("SAMPLE DATA CHAIN")
    print("=" * 60)

    # Find a model with benchmark runs
    for model in models[:20]:
        if model.benchmark_runs and len(model.benchmark_runs) > 0:
            print(f"\nModel: {model.model_id}")
            print(f"  Release date: {model.release_date}")

            # Get model group for accessibility
            if model.model_group:
                mg = model.model_group
                print(f"  Model Group: {mg.id}")
                print(f"  Accessibility: {getattr(mg, 'accessibility', 'N/A')}")
                print(f"  Training Compute: {getattr(mg, 'training_compute', 'N/A')}")

                # Get organization
                if hasattr(mg, 'organizations') and mg.organizations:
                    for org in mg.organizations[:3]:
                        print(f"  Organization: {getattr(org, 'name', org.id)}")

            # Get a benchmark run
            if model.benchmark_runs:
                run = model.benchmark_runs[0]
                print(f"\n  Sample Benchmark Run: {run.id}")
                print(f"    Status: {run.status}")
                print(f"    Started: {run.started_at}")

                # Get task
                if run.task:
                    task = run.task
                    print(f"    Task: {task.name}")
                    print(f"    Task Path: {task.path}")

                # Get scores
                if run.scores:
                    print(f"    Scores ({len(run.scores)}):")
                    for score in run.scores[:3]:
                        print(f"      - {score.scorer}: {score.mean:.4f} (stderr: {score.stderr:.4f})")

            break

    # Summary for implementation
    print("\n" + "=" * 60)
    print("RECOMMENDED BENCHMARKS FOR GAP ANALYSIS")
    print("=" * 60)
    print("\nBenchmarks with 20+ model evaluations:")
    for b in benchmark_info:
        if b['num_models'] >= 20:
            print(f"  - {b['name']} ({b['num_models']} models)")
            print(f"      Path: {b['path']}")

    return {
        'scores': scores,
        'tasks': tasks,
        'models': models,
        'model_groups': model_groups,
        'organizations': organizations,
        'runs': runs,
        'benchmark_info': benchmark_info
    }

if __name__ == "__main__":
    data = explore_data()
