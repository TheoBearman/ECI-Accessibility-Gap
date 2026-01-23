"""
Benchmark Data Fetcher Module

Fetches benchmark data from Epoch AI's Airtable database and transforms it
into the format used by the ECI accessibility gap analysis pipeline.

Supports multiple benchmarks: GPQA Diamond, MATH Level 5, SWE-Bench, etc.
"""
import os
from datetime import datetime
from collections import defaultdict
from typing import Optional

# Environment variables for Airtable access
# These should be set via environment or GitHub secrets
# AIRTABLE_BASE_ID - The Airtable base ID (starts with "app")
# AIRTABLE_PERSONAL_ACCESS_TOKEN - API token with data.records:read scope

from epochai.airtable.models import MLModel, Task, Score, BenchmarkRun, MLModelGroup


# Benchmark metadata configuration
BENCHMARK_CONFIG = {
    "gpqa_diamond": {
        "name": "GPQA Diamond",
        "task_path": "bench.task.gpqa.gpqa_diamond",
        "unit": "Accuracy",
        "scale": 100,  # Convert 0-1 to percentage
        "threshold": 1.0,  # Gap matching threshold (percentage points)
        "scorer": "choice",  # Which scorer to use for this benchmark
        "description": "Graduate-level science questions (Diamond subset)"
    },
    "math_level_5": {
        "name": "MATH Level 5",
        "task_path": "bench.task.hendrycks_math.hendrycks_math_lvl_5",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 1.0,
        "scorer": "sympy_equiv",  # Main accuracy scorer for MATH
        "description": "Competition mathematics problems (hardest level)"
    },
    "otis_mock_aime": {
        "name": "OTIS Mock AIME",
        "task_path": "bench.task.otis_mock_aime.otis_mock_aime_24_25",
        "unit": "Score",
        "scale": 100,
        "threshold": 1.0,
        "scorer": "model_extracted_exact_match",
        "description": "Mock AIME competition problems"
    },
    "swe_bench_verified": {
        "name": "SWE-Bench Verified",
        "task_path": "bench.task.swe_bench.swe_bench_verified",
        "unit": "Resolve Rate",
        "scale": 100,
        "threshold": 1.0,
        "scorer": "swe_bench_scorer",
        "description": "Software engineering bug fixing benchmark"
    },
    "simpleqa_verified": {
        "name": "SimpleQA Verified",
        "task_path": "bench.task.simpleqa.simpleqa_verified",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 1.0,
        "scorer": "simpleqa_scorer",
        "description": "Simple factual question answering"
    },
    "frontiermath_public": {
        "name": "FrontierMath (Public)",
        "task_path": "bench.task.frontiermath.frontiermath_2025_02_28_public",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 0.5,  # Lower threshold for harder benchmark
        "scorer": "verification_code",
        "description": "Frontier-level mathematics problems (public subset)"
    },
    "chess_puzzles": {
        "name": "Chess Puzzles",
        "task_path": "bench.task.chess_puzzles.chess_puzzles",
        "unit": "Accuracy",
        "scale": 100,
        "threshold": 1.0,
        "scorer": "model_extracted_exact_match",
        "description": "Chess tactical puzzles"
    }
}

# Organizations known to be Chinese
CHINA_ORGS = {
    '01.AI', 'Alibaba', 'Alibaba Cloud', 'Baidu', 'ByteDance',
    'DeepSeek', 'iFlytek', 'Huawei', 'Tencent', 'Zhipu AI',
    'Shanghai AI Laboratory', 'Baichuan', 'THUDM', 'Moonshot AI',
    'SenseTime', 'IDEA Research', 'ModelBest', 'Qwen Team'
}

# Accessibility values that indicate open weights
OPEN_ACCESSIBILITY = {
    'Open weights (unrestricted)',
    'Open weights (non-commercial)',
    'Open weights (restricted use)'
}

# Organizations known to release all models as open source
# Used as a fallback when accessibility metadata is missing
OPEN_SOURCE_ORGS = {
    'DeepSeek', 'Meta', 'Mistral', 'Alibaba', 'Alibaba Cloud',
    'Qwen Team', '01.AI', 'Zhipu AI', 'THUDM', 'Baichuan',
    'BigScience', 'EleutherAI', 'Stability AI', 'Together AI',
    'Shanghai AI Laboratory', 'IDEA Research', 'ModelBest'
}


class BenchmarkDataFetcher:
    """Fetches and transforms benchmark data from Epoch AI's Airtable."""

    def __init__(self):
        self._tasks = None
        self._runs = None
        self._scores = None
        self._models = None
        self._model_groups = None
        self._task_by_path = None
        self._model_by_id = None
        self._run_by_id = None

    def _load_data(self):
        """Load all data from Airtable with memoization."""
        if self._tasks is not None:
            return

        print("Loading data from Epoch AI Airtable...")
        self._tasks = Task.all(memoize=True)
        self._runs = BenchmarkRun.all(memoize=True)
        self._scores = Score.all(memoize=True)
        self._models = MLModel.all(memoize=True)
        self._model_groups = MLModelGroup.all(memoize=True)

        # Build lookup dictionaries
        self._task_by_path = {t.path: t for t in self._tasks}
        self._model_by_id = {m.id: m for m in self._models}
        self._run_by_id = {r.id: r for r in self._runs}

        print(f"  Loaded {len(self._tasks)} tasks, {len(self._runs)} runs, "
              f"{len(self._models)} models, {len(self._model_groups)} model groups")

    def get_available_benchmarks(self) -> list:
        """Return list of available benchmarks with their metadata."""
        self._load_data()

        available = []
        for bench_id, config in BENCHMARK_CONFIG.items():
            task = self._task_by_path.get(config["task_path"])
            if task:
                # Count models with scores for this benchmark
                model_count = self._count_models_for_task(task)
                available.append({
                    "id": bench_id,
                    "name": config["name"],
                    "description": config["description"],
                    "unit": config["unit"],
                    "model_count": model_count
                })

        return available

    def _count_models_for_task(self, task) -> int:
        """Count unique models evaluated on a task."""
        model_ids = set()
        for run in self._runs:
            if run.task and run.task.id == task.id and run.model:
                model_ids.add(run.model.id)
        return len(model_ids)

    def _get_organization_names(self, model) -> list:
        """Get all organization names for a model."""
        names = []
        if model.model_group:
            mg = model.model_group
            if hasattr(mg, 'organizations') and mg.organizations:
                for org in mg.organizations:
                    name = getattr(org, 'name', None)
                    if name:
                        names.append(name)
        return names if names else ['Unknown']

    def _get_organization_name(self, model) -> str:
        """Get the primary organization name for a model."""
        names = self._get_organization_names(model)
        return names[0] if names else 'Unknown'

    def _is_china_model(self, model) -> bool:
        """Check if any of the model's organizations are Chinese."""
        org_names = self._get_organization_names(model)
        return any(self._is_china_org(name) for name in org_names)

    def _is_open_model(self, model) -> bool:
        """Determine if a model is open weights.

        First checks the accessibility field from model_group.
        Falls back to checking if the organization is known to release open models.
        """
        # Check accessibility metadata first
        if model.model_group:
            accessibility = getattr(model.model_group, 'accessibility', None)
            if accessibility:
                return accessibility in OPEN_ACCESSIBILITY

        # Fallback: check if organization is known to release open models
        org_names = self._get_organization_names(model)
        for org_name in org_names:
            if any(open_org.lower() in org_name.lower() for open_org in OPEN_SOURCE_ORGS):
                return True

        return False

    def _is_china_org(self, org_name: str) -> bool:
        """Check if organization is Chinese."""
        if not org_name:
            return False
        return any(china_org.lower() in org_name.lower() for china_org in CHINA_ORGS)

    def fetch_benchmark_data(self, benchmark_id: str) -> Optional[dict]:
        """
        Fetch and transform data for a specific benchmark.

        Returns data in the same format as the ECI pipeline:
        {
            "models": [...],  # Model records with scores
            "metadata": {...}  # Benchmark metadata
        }
        """
        if benchmark_id not in BENCHMARK_CONFIG:
            print(f"Unknown benchmark: {benchmark_id}")
            return None

        config = BENCHMARK_CONFIG[benchmark_id]
        self._load_data()

        # Find the task
        task = self._task_by_path.get(config["task_path"])
        if not task:
            print(f"Task not found: {config['task_path']}")
            return None

        print(f"Fetching data for {config['name']}...")

        # Collect scores for each model on this task
        model_scores = defaultdict(list)

        for run in self._runs:
            if not run.task or run.task.id != task.id:
                continue
            if not run.model or run.status != 'Success':
                continue

            model = run.model

            # Get scores for this run
            if run.scores:
                for score in run.scores:
                    # Filter by scorer if specified
                    if config["scorer"] and score.scorer != config["scorer"]:
                        continue

                    model_scores[model.id].append({
                        'model': model,
                        'score': score.mean * config["scale"],
                        'stderr': (score.stderr or 0) * config["scale"],
                        'run_date': run.started_at
                    })

        # Convert to model records (using best score per model)
        models = []
        for model_id, scores in model_scores.items():
            if not scores:
                continue

            # Use the best score for this model
            best = max(scores, key=lambda x: x['score'])
            model = best['model']

            org_names = self._get_organization_names(model)
            org_display = ", ".join(org_names) if org_names else "Unknown"
            is_open = self._is_open_model(model)
            is_china = self._is_china_model(model)

            # Get release date
            release_date = model.release_date
            if release_date:
                date_str = release_date.strftime("%Y-%m-%dT%H:%M:%S")
            else:
                # Fall back to run date if no release date
                date_str = best['run_date'].strftime("%Y-%m-%dT%H:%M:%S") if best['run_date'] else None

            if not date_str:
                continue  # Skip models without dates

            models.append({
                "model": model.model_id,
                "display_name": model.model_id,
                "score": best['score'],
                "score_std": best['stderr'],
                "date": date_str,
                "organization": org_display,
                "is_open": is_open,
                "is_china": is_china
            })

        # Sort by date
        models.sort(key=lambda x: x['date'])

        print(f"  Found {len(models)} models with scores")
        print(f"  Open: {sum(1 for m in models if m['is_open'])}, "
              f"Closed: {sum(1 for m in models if not m['is_open'])}")

        return {
            "models": models,
            "metadata": {
                "id": benchmark_id,
                "name": config["name"],
                "description": config["description"],
                "unit": config["unit"],
                "threshold": config["threshold"],
                "scale": config["scale"]
            }
        }


def main():
    """Test the benchmark fetcher."""
    fetcher = BenchmarkDataFetcher()

    # List available benchmarks
    print("\n" + "=" * 60)
    print("AVAILABLE BENCHMARKS")
    print("=" * 60)

    benchmarks = fetcher.get_available_benchmarks()
    for b in benchmarks:
        print(f"\n  {b['name']} ({b['id']})")
        print(f"    {b['description']}")
        print(f"    Models: {b['model_count']}, Unit: {b['unit']}")

    # Fetch data for GPQA Diamond as a test
    print("\n" + "=" * 60)
    print("TEST: GPQA DIAMOND DATA")
    print("=" * 60)

    data = fetcher.fetch_benchmark_data("gpqa_diamond")
    if data:
        print(f"\nMetadata: {data['metadata']}")
        print(f"\nSample models:")
        for m in data['models'][:5]:
            print(f"  {m['display_name']}: {m['score']:.1f}% ({m['organization']}, "
                  f"{'Open' if m['is_open'] else 'Closed'})")


if __name__ == "__main__":
    main()
