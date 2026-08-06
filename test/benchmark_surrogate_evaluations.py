#!/usr/bin/env python
"""Benchmark time per surrogate model evaluation.

Author: OpenAI GPT-5 Codex

This script times a small set of representative waveform evaluations, stores
machine-readable JSON, writes a Markdown report with profiling appendices, and
writes PNG and self-contained HTML timing dashboards.

Typical usage:

    python test/benchmark_surrogate_evaluations.py

Compare the current checkout against a local branch or git ref:

    python test/benchmark_surrogate_evaluations.py --compare-ref my-speedup-branch --compare-label SPEEDUP

Compare against an open GitHub pull request:

    python test/benchmark_surrogate_evaluations.py --fetch-ref pull/123/head --compare-ref FETCH_HEAD --compare-label PR-123

Note: --compare-ref is the git ref that gets checked out for comparison.
--compare-label is only the display label used in the Markdown/PNG reports.
They can be the same for a local branch, but often differ for refs like
FETCH_HEAD, origin/testing, or pull-request refs.

When benchmarking a comparison ref (e.g. coming from a PR), this runner creates a temporary worktree
for that ref. If the worktree contains a compatible copy of this benchmark
script, the comparison run uses that ref-side script; otherwise it falls back
to the script from the current checkout. This lets PRs add model-specific
benchmark configuration, such as a new ENABLED_MODELS entry, while still
allowing older refs to be compared when they do not have this script yet.
If a model exists only in one run, the reports show that model with blank
timing cells for refs that cannot evaluate it.

Add another model by adding its input parameters to MODEL_CONFIGS and adding
its name to ENABLED_MODELS.
"""

from __future__ import annotations

import argparse
import cProfile
import datetime as _datetime
import html as _html
import io
import json
import math
import os
import platform
import pstats
import shutil
import subprocess
import sys
import tempfile
import timeit
from pathlib import Path
from typing import Any


PRECESSING_CONFIG = {
    "params": {
        "q": 4,
        "chiA": [-0.2, 0.4, 0.1],
        "chiB": [-0.5, 0.2, -0.4],
    },
    "dimensionless": {"dt": [0.1, 0.5], "f_low": [0, 0.01]},
    "mks": {
        "dt": [1.0 / 4096.0, 1.0 / 8192.0],
        "f_low": [0, 20],
        "extra_kwargs": {"f_ref": 20, "ellMax": None, "M": 70, "dist_mpc": 100, "units": "mks"},
    },
}

HYBRID_CONFIG = {
    "params": {
        "q": 7,
        "chiA": [0, 0, 0.5],
        "chiB": [0, 0, 0.0], # needed to work for NRHybSur2dq15
    },
    "dimensionless": {"dt": [0.1, 0.5], "f_low": [1e-2, 2e-3]},
    "mks": {
        "dt": [1.0 / 4096.0, 1.0 / 8192.0],
        "f_low": [7, 20],
        "extra_kwargs": {"f_ref": 20, "ellMax": None, "M": 70, "dist_mpc": 100, "units": "mks"},
    },
}

PRECESSING_MODELS = ["NRSur7dq4", "SEOBNRv4PHMSur"]
HYBRID_MODELS = ["NRHybSur3dq8", "NRHybSur2dq15", "NRHybSur3dq8_CCE"]

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    **{model: PRECESSING_CONFIG for model in PRECESSING_MODELS},
    **{model: HYBRID_CONFIG for model in HYBRID_MODELS},
}
ENABLED_MODELS = PRECESSING_MODELS + HYBRID_MODELS

BLAS_THREADING_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "GOTO_NUM_THREADS",
    "OMP_DYNAMIC",
    "MKL_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_CPUS_PER_NODE",
]


def dimensionless_cases(dt_values: list[float], f_low_values: list[float]) -> list[dict[str, Any]]:
    """Build the geometric-unit benchmark cases."""
    cases = []
    for dt in dt_values:
        for f_low in f_low_values:
            cases.append(
                {
                    "id": f"geom_dt_{dt:g}_flow_{f_low:g}",
                    "group": "Geometric Units",
                    "header": f"dt = {dt:g}M<br>f_low = {f_low:g}",
                    "kwargs": {"dt": dt, "f_low": f_low},
                }
            )
    return cases


def mks_cases(dt_values: list[float], f_low_values: list[float], extra_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the MKS benchmark cases."""
    cases = []
    for dt in dt_values:
        for f_low in f_low_values:
            cases.append(
                {
                    "id": f"mks_dt_{dt:.12g}_flow_{f_low:g}",
                    "group": "MKS Units (M_tot = 70 M_sun)",
                    "header": f"dt = 1/{round(1.0 / dt):g}s<br>f_low = {f_low:g} Hz",
                    "kwargs": {
                        "dt": dt,
                        "f_low": f_low,
                        **extra_kwargs,
                    },
                }
            )
    return cases


def model_cases(model: str) -> list[dict[str, Any]]:
    """Build benchmark cases for a model from MODEL_CONFIGS."""
    config = model_config(model)
    mks_config = config["mks"]
    dimensionless_config = config["dimensionless"]
    return (
        mks_cases(mks_config["dt"], mks_config["f_low"], mks_config["extra_kwargs"])
        + dimensionless_cases(dimensionless_config["dt"], dimensionless_config["f_low"])
    )


def benchmark_models(gws: Any) -> list[str]:
    """Return enabled benchmark models after validating configuration."""
    catalog_models = set(gws.catalog._surrogate_world)
    missing_from_catalog = [model for model in ENABLED_MODELS if model not in catalog_models]
    missing_configs = [model for model in ENABLED_MODELS if model not in MODEL_CONFIGS]
    if missing_from_catalog:
        raise ValueError(f"Enabled models missing from catalog: {missing_from_catalog}")
    if missing_configs:
        raise ValueError(f"Enabled models missing from MODEL_CONFIGS: {missing_configs}")
    return list(ENABLED_MODELS)


def run_command(args: list[str], cwd: Path | None = None, check: bool = False) -> str:
    """Run a command and return combined stdout/stderr as stripped text."""
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd) if cwd else None,
            check=check,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return str(exc)
    return completed.stdout.strip()


def update_submodules(worktree: Path) -> None:
    """Initialize submodules at the commits recorded by a temporary worktree."""
    print(f"Updating submodules in {worktree}", flush=True)
    subprocess.run(
        [
            "git",
            "-c",
            "url.https://github.com/.insteadOf=git@github.com:",
            "-c",
            "url.https://github.com/.insteadOf=ssh://git@github.com/",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
        cwd=str(worktree),
        check=True,
    )


def symlink_surrogate_downloads(source_root: Path, destination_root: Path) -> None:
    """Symlink downloaded surrogate data into a temporary worktree."""
    relative_path = Path("gwsurrogate") / "surrogate_downloads"
    source = source_root / relative_path
    destination = destination_root / relative_path
    if not source.exists():
        return
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        try:
            destination.rmdir()
        except OSError:
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source.resolve(), target_is_directory=True)


def build_runtime_artifacts(worktree: Path) -> None:
    """Build C extension modules in place inside a temporary worktree."""
    print(f"Building C extensions in {worktree}", flush=True)
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=str(worktree),
        check=True,
    )


def git_value(args: list[str], repo_root: Path) -> str:
    """Run a git command in repo_root and return a nonempty value or unknown."""
    value = run_command(["git", *args], cwd=repo_root)
    return value if value else "unknown"


def collect_submodule_context(repo_root: Path) -> dict[str, dict[str, str]]:
    """Collect submodule commits recorded in the benchmark checkout."""
    output = run_command(["git", "submodule", "status", "--recursive"], cwd=repo_root)
    submodules: dict[str, dict[str, str]] = {}
    if not output or output == "unknown":
        return submodules

    status_labels = {
        " ": "initialized",
        "-": "not initialized",
        "+": "checked out at different commit than recorded",
        "U": "merge conflict",
    }
    for line in output.splitlines():
        if not line:
            continue
        if line[0] in status_labels:
            status_prefix = line[0]
            parts = line[1:].strip().split(maxsplit=2)
        else:
            status_prefix = " "
            parts = line.strip().split(maxsplit=2)
        if len(parts) < 2:
            continue
        commit, path = parts[:2]
        submodules[path] = {
            "status": status_labels.get(status_prefix, f"unknown ({status_prefix})"),
            "commit": commit,
        }
        if len(parts) == 3:
            submodules[path]["description"] = parts[2]
    return submodules


def collect_threading_context() -> dict[str, Any]:
    """Collect BLAS and thread-pool context for reproducible timings."""
    context: dict[str, Any] = {
        "environment": {
            name: os.environ.get(name)
            for name in BLAS_THREADING_ENV_VARS
        },
    }
    if hasattr(os, "sched_getaffinity"):
        context["sched_getaffinity_count"] = len(os.sched_getaffinity(0))

    try:
        from threadpoolctl import threadpool_info
    except ImportError as exc:
        context["threadpoolctl"] = {
            "available": False,
            "error": str(exc),
        }
    else:
        context["threadpoolctl"] = {
            "available": True,
            "info": threadpool_info(),
        }
    return context


def collect_context(repo_root: Path) -> dict[str, Any]:
    """Collect reproducibility context for a benchmark run."""
    context: dict[str, Any] = {
        "timestamp_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "python": {
            "executable": sys.executable,
            "version": sys.version.replace("\n", " "),
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "cpu_count": os.cpu_count(),
        "conda": {
            "prefix": os.environ.get("CONDA_PREFIX"),
            "default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_exe": os.environ.get("CONDA_PYTHON_EXE"),
        },
        "threading": collect_threading_context(),
        "git": {
            "repo_root": str(repo_root),
            "branch": git_value(["branch", "--show-current"], repo_root),
            "commit": git_value(["rev-parse", "HEAD"], repo_root),
            "describe": git_value(["describe", "--always", "--dirty", "--tags"], repo_root),
            "status_short": git_value(["status", "--short"], repo_root),
            "submodules": collect_submodule_context(repo_root),
        },
    }
    if shutil.which("conda"):
        context["conda"]["list"] = run_command(["conda", "list"])
        context["conda"]["env_export"] = run_command(["conda", "env", "export"])
    if shutil.which("lscpu"):
        context["hardware"] = {"lscpu": run_command(["lscpu"])}
    elif shutil.which("sysctl"):
        context["hardware"] = {"sysctl_machdep_cpu": run_command(["sysctl", "-a"])}
    return context


def model_config(model: str) -> dict[str, Any]:
    """Return the configured intrinsic parameters for a model."""
    try:
        return MODEL_CONFIGS[model]
    except KeyError as exc:
        known = ", ".join(sorted(MODEL_CONFIGS))
        raise ValueError(f"No input parameters configured for model {model!r}. Known models: {known}") from exc


def evaluate_case(sur: Any, config: dict[str, Any], case: dict[str, Any]) -> Any:
    """Evaluate one surrogate/input-case combination."""
    params = config["params"]
    return sur(params["q"], params["chiA"], params["chiB"], **case["kwargs"])


def profile_case(sur: Any, config: dict[str, Any], case: dict[str, Any], profile_limit: int) -> str:
    """Run cProfile for one surrogate/input-case combination."""
    profiler = cProfile.Profile()
    profiler.enable()
    evaluate_case(sur, config, case)
    profiler.disable()
    output = io.StringIO()
    stats = pstats.Stats(profiler, stream=output).strip_dirs().sort_stats("cumtime")
    stats.print_stats(profile_limit)
    return output.getvalue()


def run_single_benchmark(args: argparse.Namespace, repo_root: Path) -> dict[str, Any]:
    """Run all configured benchmark cases in the current Python process."""
    import gwsurrogate as gws

    models = benchmark_models(gws)
    results: list[dict[str, Any]] = []

    for model in models:
        config = model_config(model)
        cases = model_cases(model)
        print(f"Loading surrogate {model}", flush=True)
        sur = gws.LoadSurrogate(model)
        for case in cases:
            print(f"Timing {model} / {case['id']}", flush=True)
            evaluate_case(sur, config, case)
            timer = timeit.Timer(lambda: evaluate_case(sur, config, case))
            repeats = timer.repeat(repeat=args.repeat, number=args.number)
            per_eval = [elapsed / args.number for elapsed in repeats]
            profile = profile_case(sur, config, case, args.profile_limit)
            results.append(
                {
                    "model": model,
                    "case_id": case["id"],
                    "case_group": case["group"],
                    "case_header": case["header"],
                    "input": {
                        **config["params"],
                        **case["kwargs"],
                    },
                    "timing_seconds": {
                        "repeat": args.repeat,
                        "number": args.number,
                        "per_eval_repeats": per_eval,
                        "best": min(per_eval),
                        "median": sorted(per_eval)[len(per_eval) // 2],
                    },
                    "profile_cumulative": profile,
                }
            )

    return {
        "schema_version": 1,
        "label": args.run_label,
        "models": models,
        "model_configs": {model: model_config(model) for model in models},
        "cases_by_model": {model: model_cases(model) for model in models},
        "settings": {
            "repeat": args.repeat,
            "number": args.number,
            "profile": True,
            "profile_limit": args.profile_limit,
        },
        "script_source": {
            "kind": args.script_source_kind,
            "ref": args.script_source_ref,
            "path": args.script_source_path,
        },
        "context": collect_context(repo_root),
        "results": results,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write JSON data to path, creating parent directories as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    """Read JSON data from path."""
    return json.loads(path.read_text(encoding="utf-8"))


def format_seconds(value: float | None) -> str:
    """Format a timing value for compact report display."""
    if value is None:
        return ""
    return f"{value:.6g}"


def escape_html(value: Any) -> str:
    """Escape a value for safe insertion into the static HTML report."""
    return _html.escape(str(value), quote=True)


def result_lookup(run: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (model, case_id) pairs to timing result dictionaries."""
    return {
        (result["model"], result["case_id"]): result
        for result in run["results"]
    }


def case_timing(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return timing data for a result, or an empty dictionary if absent."""
    if not result:
        return {}
    return result.get("timing_seconds", {})


def cases_for_model(run: dict[str, Any], model: str) -> list[dict[str, Any]]:
    """Return the benchmark cases associated with model in a run."""
    if "cases_by_model" in run:
        return run["cases_by_model"].get(model, [])
    return run.get("cases", [])


def cases_for_model_across_runs(runs: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    """Return cases for a model from the first run that includes them."""
    for run in runs:
        cases = cases_for_model(run, model)
        if cases:
            return cases
    return []


def case_dt_label(case: dict[str, Any]) -> str:
    """Return a display label for a benchmark case timestep."""
    kwargs = case["kwargs"]
    if kwargs.get("units") == "mks":
        return f"1/{round(1.0 / kwargs['dt']):g} s"
    return f"{kwargs['dt']:g} M"


def case_f_low_label(case: dict[str, Any]) -> str:
    """Return a display label for a benchmark case low frequency."""
    kwargs = case["kwargs"]
    if kwargs.get("units") == "mks":
        return f"{kwargs['f_low']:g} Hz"
    return f"{kwargs['f_low']:g}"


def format_speedup(baseline: float | None, comparison: float | None) -> str:
    """Format a speedup ratio from two timing values."""
    if baseline is None or comparison in (None, 0):
        return ""
    return f"{baseline / comparison:.3g}x"


def model_parameter_label(model: str, benchmark: dict[str, Any]) -> str:
    """Return a concise label describing the configured model parameters."""
    runs = benchmark.get("runs", [benchmark])
    config = None
    for run in runs:
        config = run.get("model_configs", {}).get(model)
        if config:
            break
    if config is None:
        try:
            config = model_config(model)
        except ValueError:
            return f"{model}: parameters unavailable"
    params = config.get("params", config)
    mks_kwargs = config.get("mks", {}).get("extra_kwargs", {})
    ell_max = mks_kwargs.get("ellMax")
    return (
        f"{model}: q={params['q']}, chiA={params['chiA']}, "
        f"chiB={params['chiB']}, ellMax={ell_max}"
    )


def png_table_data(benchmark: dict[str, Any]) -> list[list[str]]:
    """Build PNG table rows with grouped run timing columns."""
    rows = []
    runs = benchmark.get("runs", [benchmark])
    run_lookups = [result_lookup(run) for run in runs]
    models = list(dict.fromkeys(model for run in runs for model in run["models"]))
    _, header_bottom = png_table_headers(benchmark)
    column_count = len(header_bottom)

    for model in models:
        rows.append([model_parameter_label(model, benchmark), *[""] * (column_count - 1)])
        cases = cases_for_model_across_runs(runs, model)
        for case in cases:
            row = [model, case["group"], case_dt_label(case), case_f_low_label(case)]
            run_timings = [
                case_timing(lookup.get((model, case["id"])))
                for lookup in run_lookups
            ]
            for timing in run_timings:
                row.extend([format_seconds(timing.get("best")), format_seconds(timing.get("median"))])
            baseline_best = run_timings[0].get("best") if run_timings else None
            for timing in run_timings[1:]:
                row.append(format_speedup(baseline_best, timing.get("best")))
            rows.append(row)
    return rows


def png_table_headers(benchmark: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Build the two-row grouped header for the PNG timing table."""
    runs = benchmark.get("runs", [benchmark])
    top = ["model", "units", "dt", "f_low"]
    bottom = ["", "", "", ""]
    for run in runs:
        top.extend(["", ""])
        bottom.extend(["best (s)", "median (s)"])
    if len(runs) == 2:
        top.append("speedup")
        bottom.append(f"{runs[0]['label']} / {runs[1]['label']}")
    elif len(runs) > 2:
        baseline = runs[0]["label"]
        for run in runs[1:]:
            top.append(f"speedup vs {baseline}")
            bottom.append(run["label"])
    return top, bottom


def is_model_parameter_row(row: list[str]) -> bool:
    """Return whether a PNG table row is a model-parameter separator row."""
    return bool(row and ": q=" in row[0] and "chiA=" in row[0] and "chiB=" in row[0])


def draw_grouped_run_labels(ax: Any, fig: Any, table: Any, runs: list[dict[str, Any]]) -> None:
    """Draw run labels centered over each best/median column pair."""
    from matplotlib.patches import Rectangle

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for run_index, run in enumerate(runs):
        left_column = 4 + 2 * run_index
        right_column = left_column + 1
        left_bbox = table[(0, left_column)].get_window_extent(renderer)
        right_bbox = table[(0, right_column)].get_window_extent(renderer)
        x0_axes, y0_axes = ax.transAxes.inverted().transform((left_bbox.x0, left_bbox.y0))
        x1_axes, y1_axes = ax.transAxes.inverted().transform((right_bbox.x1, right_bbox.y1))
        ax.add_patch(
            Rectangle(
                (x0_axes, y0_axes),
                x1_axes - x0_axes,
                y1_axes - y0_axes,
                facecolor="#d9ead3",
                edgecolor="black",
                linewidth=1,
                transform=ax.transAxes,
                zorder=5,
            )
        )
        ax.text(
            (x0_axes + x1_axes) / 2.0,
            (y0_axes + y1_axes) / 2.0,
            run["label"],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=6,
        )


def draw_model_parameter_rows(ax: Any, fig: Any, table: Any, rows: list[list[str]]) -> None:
    """Draw full-width model-parameter rows over the table cells."""
    from matplotlib.patches import Rectangle

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    last_column = len(rows[0]) - 1
    for row_index, row in enumerate(rows):
        if not is_model_parameter_row(row):
            continue
        left_bbox = table[(row_index, 0)].get_window_extent(renderer)
        right_bbox = table[(row_index, last_column)].get_window_extent(renderer)
        x0_axes, y0_axes = ax.transAxes.inverted().transform((left_bbox.x0, left_bbox.y0))
        x1_axes, y1_axes = ax.transAxes.inverted().transform((right_bbox.x1, right_bbox.y1))
        ax.add_patch(
            Rectangle(
                (x0_axes, y0_axes),
                x1_axes - x0_axes,
                y1_axes - y0_axes,
                facecolor="#fff2cc",
                edgecolor="black",
                linewidth=1,
                transform=ax.transAxes,
                zorder=5,
            )
        )
        ax.text(
            (x0_axes + x1_axes) / 2.0,
            (y0_axes + y1_axes) / 2.0,
            row[0],
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            transform=ax.transAxes,
            zorder=6,
        )


def write_markdown(path: Path, benchmark: dict[str, Any], png_path: Path | None = None) -> None:
    """Write the human-readable Markdown benchmark report."""
    runs = benchmark.get("runs", [benchmark])

    lines = [
        "# GWSurrogate Evaluation Timing",
        "",
        f"Generated: {benchmark['generated_utc']}",
        "",
        "Times below are seconds per model evaluation. Raw repeats and context are in the JSON output.",
    ]
    if png_path is not None:
        lines.extend(["", f"PNG timing table: `{png_path}`"])
    lines.extend(["", "## Summary", ""])
    for run in runs:
        by_key = result_lookup(run)
        lines.extend([f"### {run['label']}", ""])
        for model in run["models"]:
            cases = cases_for_model(run, model)
            case_groups = list(dict.fromkeys(case["group"] for case in cases))
            lines.extend([f"#### {model}", ""])
            for group in case_groups:
                group_cases = [case for case in cases if case["group"] == group]
                lines.extend([f"{group}:", ""])
                for case in group_cases:
                    timing = case_timing(by_key.get((model, case["id"])))
                    lines.append(
                        "- "
                        f"`dt={case_dt_label(case)}`, `f_low={case_f_low_label(case)}`: "
                        f"best `{format_seconds(timing.get('best'))}` s, "
                        f"median `{format_seconds(timing.get('median'))}` s"
                    )
                lines.append("")

    lines.extend(["## Context", ""])
    for run in runs:
        context = run["context"]
        git = context.get("git", {})
        platform_info = context.get("platform", {})
        conda = context.get("conda", {})
        lines.extend(
            [
                f"### {run['label']}",
                "",
                f"- Git branch: `{git.get('branch', 'unknown')}`",
                f"- Git commit: `{git.get('commit', 'unknown')}`",
                f"- Git describe: `{git.get('describe', 'unknown')}`",
                f"- Python: `{context.get('python', {}).get('version', 'unknown')}`",
                f"- Platform: `{platform_info.get('system', '')} {platform_info.get('release', '')} {platform_info.get('machine', '')}`",
                f"- CPU count: `{context.get('cpu_count', 'unknown')}`",
                f"- Conda env: `{conda.get('default_env') or conda.get('prefix') or 'unknown'}`",
                "",
            ]
        )
        status = git.get("status_short")
        if status and status != "unknown":
            lines.extend(["Git status:", "", "```text", status, "```", ""])
        submodules = git.get("submodules", {})
        if submodules:
            lines.extend(["Submodules:", ""])
            for submodule_path, submodule in sorted(submodules.items()):
                description = submodule.get("description")
                detail = f" ({description})" if description else ""
                lines.append(
                    "- "
                    f"`{submodule_path}`: `{submodule.get('commit', 'unknown')}` "
                    f"{submodule.get('status', 'unknown')}{detail}"
                )
            lines.append("")

    lines.extend(["## Appendix", "", "### Hardware Data", ""])
    for run in runs:
        hardware = run["context"].get("hardware", {})
        lines.extend([f"#### {run['label']}", ""])
        if not hardware:
            lines.extend(["No hardware data recorded.", ""])
            continue
        for key, value in hardware.items():
            lines.extend([f"{key}:", "", "```text", value, "```", ""])

    lines.extend(["### cProfile", ""])
    for run in runs:
        lines.extend([f"#### {run['label']}", ""])
        for result in run["results"]:
            profile = result.get("profile_cumulative")
            if not profile:
                continue
            title = f"{result['model']} / {result['case_id']}"
            lines.extend([f"##### {title}", "", "```text", profile.rstrip(), "```", ""])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def geometric_mean(values: list[float]) -> float | None:
    """Return the geometric mean of positive values, or None for no data."""
    positive_values = [value for value in values if value > 0]
    if not positive_values:
        return None
    return math.exp(sum(math.log(value) for value in positive_values) / len(positive_values))


def speedup_class(value: float | None) -> str:
    """Return the CSS class for a speedup value."""
    if value is None:
        return "neutral"
    if value >= 1.05:
        return "faster"
    if value <= 0.95:
        return "slower"
    return "neutral"


def html_run_summary(run: dict[str, Any]) -> str:
    """Render a compact run context summary for the HTML dashboard."""
    context = run["context"]
    git = context.get("git", {})
    platform_info = context.get("platform", {})
    conda = context.get("conda", {})
    items = [
        ("branch", git.get("branch", "unknown")),
        ("commit", git.get("commit", "unknown")),
        ("describe", git.get("describe", "unknown")),
        ("python", context.get("python", {}).get("version", "unknown")),
        ("platform", f"{platform_info.get('system', '')} {platform_info.get('release', '')} {platform_info.get('machine', '')}"),
        ("cpu count", context.get("cpu_count", "unknown")),
        ("conda env", conda.get("default_env") or conda.get("prefix") or "unknown"),
    ]
    for submodule_path, submodule in sorted(git.get("submodules", {}).items()):
        description = submodule.get("description")
        detail = f" ({description})" if description else ""
        items.append(
            (
                f"submodule {submodule_path}",
                f"{submodule.get('commit', 'unknown')} {submodule.get('status', 'unknown')}{detail}",
            )
        )
    rows = "\n".join(
        f"<tr><th>{escape_html(label)}</th><td>{escape_html(value)}</td></tr>"
        for label, value in items
    )
    return f"""
      <section class="panel">
        <h3>{escape_html(run['label'])}</h3>
        <table class="meta"><tbody>{rows}</tbody></table>
      </section>
    """


def html_speedup_overview(benchmark: dict[str, Any]) -> str:
    """Render per-model geometric mean speedups for two-run comparisons."""
    runs = benchmark.get("runs", [benchmark])
    if len(runs) < 2:
        return ""
    baseline, comparison = runs[0], runs[1]
    baseline_lookup = result_lookup(baseline)
    comparison_lookup = result_lookup(comparison)
    cards = []
    for model in baseline["models"]:
        ratios = []
        for case in cases_for_model(baseline, model):
            baseline_best = case_timing(baseline_lookup.get((model, case["id"]))).get("best")
            comparison_best = case_timing(comparison_lookup.get((model, case["id"]))).get("best")
            if baseline_best and comparison_best:
                ratios.append(baseline_best / comparison_best)
        speedup = geometric_mean(ratios)
        label = f"{speedup:.3g}x" if speedup else ""
        cards.append(
            f"""
            <div class="metric {speedup_class(speedup)}">
              <div class="metric-label">{escape_html(model)}</div>
              <div class="metric-value">{escape_html(label)}</div>
              <div class="metric-note">geomean {escape_html(baseline['label'])} / {escape_html(comparison['label'])}</div>
            </div>
            """
        )
    return f"""
      <section>
        <h2>Speedup Overview</h2>
        <div class="metrics">{''.join(cards)}</div>
      </section>
    """


def html_model_table(benchmark: dict[str, Any], model: str) -> str:
    """Render the HTML timing table for one model."""
    runs = benchmark.get("runs", [benchmark])
    run_lookups = [result_lookup(run) for run in runs]
    cases = cases_for_model_across_runs(runs, model)
    parameter_label = model_parameter_label(model, benchmark)
    header_top = ["<th rowspan=\"2\">units</th>", "<th rowspan=\"2\">dt</th>", "<th rowspan=\"2\">f_low</th>"]
    header_bottom = []
    for run in runs:
        header_top.append(f"<th colspan=\"2\">{escape_html(run['label'])}</th>")
        header_bottom.extend(["<th>best (s)</th>", "<th>median (s)</th>"])
    if len(runs) == 2:
        header_top.append("<th rowspan=\"2\">speedup</th>")
    body_rows = []
    for case in cases:
        cells = [
            f"<td>{escape_html(case['group'])}</td>",
            f"<td>{escape_html(case_dt_label(case))}</td>",
            f"<td>{escape_html(case_f_low_label(case))}</td>",
        ]
        run_timings = [
            case_timing(lookup.get((model, case["id"])))
            for lookup in run_lookups
        ]
        for timing in run_timings:
            cells.extend(
                [
                    f"<td class=\"num\">{escape_html(format_seconds(timing.get('best')))}</td>",
                    f"<td class=\"num\">{escape_html(format_seconds(timing.get('median')))}</td>",
                ]
            )
        if len(runs) == 2:
            baseline_best = run_timings[0].get("best")
            comparison_best = run_timings[1].get("best")
            speedup = baseline_best / comparison_best if baseline_best and comparison_best else None
            width = min(100.0, 50.0 * speedup) if speedup else 0.0
            cells.append(
                f"""
                <td class="speed {speedup_class(speedup)}">
                  <span>{escape_html(format_speedup(baseline_best, comparison_best))}</span>
                  <div class="bar"><i style="width:{width:.1f}%"></i></div>
                </td>
                """
            )
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return f"""
      <section class="model">
        <h3>{escape_html(model)}</h3>
        <div class="params">{escape_html(parameter_label)}</div>
        <div class="table-wrap">
          <table class="timings">
            <thead>
              <tr>{''.join(header_top)}</tr>
              <tr>{''.join(header_bottom)}</tr>
            </thead>
            <tbody>{''.join(body_rows)}</tbody>
          </table>
        </div>
      </section>
    """


def html_hardware_data(benchmark: dict[str, Any]) -> str:
    """Render hardware data in collapsible HTML details blocks."""
    runs = benchmark.get("runs", [benchmark])
    blocks = []
    for run in runs:
        hardware = run["context"].get("hardware", {})
        if not hardware:
            body = "<p class=\"small\">No hardware data recorded.</p>"
        else:
            body = "".join(
                f"""
                <details>
                  <summary>{escape_html(key)}</summary>
                  <pre>{escape_html(value)}</pre>
                </details>
                """
                for key, value in hardware.items()
            )
        blocks.append(
            f"""
            <section class="panel">
              <h4>{escape_html(run['label'])}</h4>
              {body}
            </section>
            """
        )
    return f"<section><h3>Hardware Data</h3><div class=\"grid\">{''.join(blocks)}</div></section>"


def html_profiles(benchmark: dict[str, Any]) -> str:
    """Render cProfile output in collapsible HTML details blocks."""
    runs = benchmark.get("runs", [benchmark])
    blocks = []
    for run in runs:
        run_blocks = []
        for result in run["results"]:
            profile = result.get("profile_cumulative")
            if not profile:
                continue
            title = f"{result['model']} / {result['case_id']}"
            run_blocks.append(
                f"""
                <details>
                  <summary>{escape_html(title)}</summary>
                  <pre>{escape_html(profile.rstrip())}</pre>
                </details>
                """
            )
        blocks.append(
            f"""
            <section class="panel">
              <h3>{escape_html(run['label'])}</h3>
              {''.join(run_blocks)}
            </section>
            """
        )
    return f"<section><h3>cProfile</h3>{''.join(blocks)}</section>"


def html_appendix(benchmark: dict[str, Any]) -> str:
    """Render the HTML appendix sections."""
    return f"""
      <section>
        <h2>Appendix</h2>
        {html_hardware_data(benchmark)}
        {html_profiles(benchmark)}
      </section>
    """


def html_script_sources(benchmark: dict[str, Any]) -> str:
    """Render the benchmark-script source metadata for each run."""
    runs = benchmark.get("runs", [benchmark])
    rows = []
    for run in runs:
        source = run.get("script_source", {})
        kind = source.get("kind", "current checkout")
        ref = source.get("ref") or run.get("label", "unknown")
        path = source.get("path", "test/benchmark_surrogate_evaluations.py")
        rows.append(
            "<tr>"
            f"<th>{escape_html(run['label'])}</th>"
            f"<td>{escape_html(kind)}</td>"
            f"<td>{escape_html(ref)}</td>"
            f"<td><code>{escape_html(path)}</code></td>"
            "</tr>"
        )
    return f"""
      <div class="notice script-source">
        <strong>Benchmark script sources</strong>
        <p>The table below shows which checkout supplied the benchmark script for each run.</p>
        <table class="notice-table">
          <thead>
            <tr><th>run</th><th>script source</th><th>ref</th><th>path</th></tr>
          </thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    """


def write_html_dashboard(path: Path, benchmark: dict[str, Any]) -> None:
    """Write a self-contained static HTML benchmark dashboard."""
    runs = benchmark.get("runs", [benchmark])
    models = list(dict.fromkeys(model for run in runs for model in run["models"]))
    settings = runs[0].get("settings", {})
    css = """
      :root { color-scheme: light; --ink:#17202a; --muted:#5f6b7a; --line:#d7dee8; --panel:#ffffff; --bg:#f5f7fb; --green:#d9ead3; --yellow:#fff2cc; --red:#f8d7da; --blue:#d9eaf7; }
      * { box-sizing: border-box; }
      body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
      header { padding: 28px 36px; background: #1f2937; color: white; }
      header h1 { margin: 0 0 8px; font-size: 28px; }
      header p { margin: 0; color: #d1d5db; }
      main { padding: 28px 36px 48px; max-width: 1500px; margin: 0 auto; }
      h2 { margin: 28px 0 12px; font-size: 22px; }
      h3 { margin: 0 0 10px; font-size: 18px; }
      .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 16px; }
      .panel, .model { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }
      .notices { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; margin-bottom: 18px; }
      .notice { border: 2px solid #d6a500; background: #fff8d6; border-radius: 8px; padding: 14px 16px; }
      .notice.hardware { border-color: #d9534f; background: #fff1f1; }
      .notice.script-source { border-color: #78a3c9; background: #edf6ff; }
      .notice strong { display: block; margin-bottom: 6px; }
      .notice p { margin: 6px 0 0; }
      .notice-table { margin-top: 10px; font-size: 13px; }
      .notice-table th, .notice-table td { background: rgba(255,255,255,.45); padding: 6px 8px; }
      .warning-text { color: #b42318; font-weight: 700; }
      .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
      .metric { border: 1px solid var(--line); border-radius: 8px; padding: 14px; background: white; }
      .metric.faster { border-color: #8bc58a; background: #f1faef; }
      .metric.slower { border-color: #e0969c; background: #fff3f3; }
      .metric-label { color: var(--muted); font-size: 13px; }
      .metric-value { font-size: 26px; font-weight: 700; margin-top: 4px; }
      .metric-note { color: var(--muted); font-size: 12px; margin-top: 2px; }
      .params { background: var(--yellow); border: 1px solid #e5cf84; border-radius: 6px; padding: 8px 10px; margin-bottom: 12px; font-weight: 600; }
      .table-wrap { overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; vertical-align: middle; }
      th { background: var(--green); font-weight: 700; text-align: center; }
      td.num { text-align: right; font-variant-numeric: tabular-nums; }
      tbody tr:nth-child(even) td { background: #f8fafc; }
      .speed { min-width: 130px; font-variant-numeric: tabular-nums; }
      .speed span { font-weight: 700; }
      .speed.faster span { color: #256029; }
      .speed.slower span { color: #8a1c25; }
      .bar { height: 7px; background: #e5e7eb; border-radius: 99px; margin-top: 5px; overflow: hidden; }
      .bar i { display: block; height: 100%; background: #60a5fa; }
      .speed.faster .bar i { background: #65a765; }
      .speed.slower .bar i { background: #d66b75; }
      .meta th { width: 120px; text-align: left; background: #eef4ee; }
      details { border: 1px solid var(--line); border-radius: 6px; margin: 8px 0; background: white; }
      summary { cursor: pointer; padding: 9px 11px; font-weight: 600; }
      pre { margin: 0; padding: 12px; overflow-x: auto; background: #111827; color: #e5e7eb; font-size: 12px; line-height: 1.45; }
      .small { color: var(--muted); font-size: 13px; }
    """
    model_sections = "\n".join(html_model_table(benchmark, model) for model in models)
    run_summaries = "\n".join(html_run_summary(run) for run in runs)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GWSurrogate Performance Benchmark</title>
  <style>{css}</style>
</head>
<body>
  <header>
    <h1>GWSurrogate Performance Benchmark</h1>
    <p>Generated {escape_html(benchmark['generated_utc'])}. Times are seconds per model evaluation.</p>
  </header>
  <main>
    <section class="notices">
      <div class="notice hardware">
        <strong>Hardware-dependent timing data</strong>
        <p>These timings depend on CPU, memory, operating system, Python, compiled libraries, and system load.</p>
        <p class="warning-text">Benchmarks run on GitHub Actions are expected to be slower and noisier than benchmarks run on dedicated HPC resources.</p>
      </div>
      <div class="notice">
        <strong>Speedup formula</strong>
        <p>Speedup is computed from best times as <code>baseline best time / comparison best time</code>.</p>
        <p>Values above <code>1x</code> mean the comparison run is faster than the baseline.</p>
      </div>
      {html_script_sources(benchmark)}
    </section>
    <section class="grid">
      <div class="panel"><h3>Benchmark Settings</h3><p class="small">repeat={escape_html(settings.get('repeat', 'unknown'))}, number={escape_html(settings.get('number', 'unknown'))}, profile_limit={escape_html(settings.get('profile_limit', 'unknown'))}</p></div>
      <div class="panel"><h3>Runs</h3><p class="small">{escape_html(', '.join(run['label'] for run in runs))}</p></div>
      <div class="panel"><h3>Models</h3><p class="small">{escape_html(', '.join(models))}</p></div>
    </section>
    {html_speedup_overview(benchmark)}
    <section><h2>Timing Tables</h2>{model_sections}</section>
    <section><h2>Run Context</h2><div class="grid">{run_summaries}</div></section>
    {html_appendix(benchmark)}
  </main>
</body>
</html>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def write_png_table(path: Path, benchmark: dict[str, Any]) -> None:
    """Write a PNG table summarizing benchmark timing results."""
    os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="gwsurrogate-mpl-"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = benchmark.get("runs", [benchmark])
    header_top, header_bottom = png_table_headers(benchmark)
    rows = [header_top, header_bottom] + png_table_data(benchmark)
    if len(rows) == 2:
        rows.append([""] * len(header_top))

    fig_width = max(13, 1.15 * len(header_top))
    fig_height = max(2.5, 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, _column), cell in table.get_celld().items():
        if row in (0, 1):
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#d9ead3")
        elif is_model_parameter_row(rows[row]):
            cell.get_text().set_text("")
            cell.set_facecolor("#fff2cc")
        elif row % 2 == 0:
            cell.set_facecolor("#f6f8fa")
    draw_grouped_run_labels(ax, fig, table, runs)
    draw_model_parameter_rows(ax, fig, table, rows)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def compatible_benchmark_script(script_path: Path) -> bool:
    """Return whether a benchmark script supports this runner's hidden CLI."""
    if not script_path.exists():
        return False
    text = script_path.read_text(encoding="utf-8")
    required_flags = ["--single-run", "--repo-root", "--json-output", "--html-output"]
    return all(flag in text for flag in required_flags)


def run_subprocess_for_ref(
    script_path: Path,
    repo_root: Path,
    ref: str,
    label: str,
    args: argparse.Namespace,
    temp_root: Path,
) -> dict[str, Any]:
    """Benchmark a git ref in a temporary worktree and return its JSON result."""
    worktree = temp_root / label.replace("/", "_")
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(worktree), ref],
        cwd=str(repo_root),
        check=True,
    )
    update_submodules(worktree)
    symlink_surrogate_downloads(repo_root, worktree)
    build_runtime_artifacts(worktree)
    try:
        try:
            relative_script_path = script_path.relative_to(repo_root)
        except ValueError:
            relative_script_path = Path("test") / "benchmark_surrogate_evaluations.py"
        ref_script_path = worktree / relative_script_path
        command_script_path = ref_script_path if compatible_benchmark_script(ref_script_path) else script_path
        if command_script_path == ref_script_path:
            script_source_kind = "comparison ref worktree"
            script_source_path = str(relative_script_path)
        else:
            script_source_kind = "current checkout fallback"
            script_source_path = str(script_path)
        json_path = temp_root / f"{label.replace('/', '_')}.json"
        md_path = temp_root / f"{label.replace('/', '_')}.md"
        png_path = temp_root / f"{label.replace('/', '_')}.png"
        command = [
            sys.executable,
            str(command_script_path),
            "--single-run",
            "--repo-root",
            str(worktree),
            "--run-label",
            label,
            "--json-output",
            str(json_path),
            "--md-output",
            str(md_path),
            "--png-output",
            str(png_path),
            "--html-output",
            str(temp_root / f"{label.replace('/', '_')}.html"),
            "--script-source-kind",
            script_source_kind,
            "--script-source-ref",
            label,
            "--script-source-path",
            script_source_path,
            "--repeat",
            str(args.repeat),
            "--number",
            str(args.number),
            "--profile-limit",
            str(args.profile_limit),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(worktree) + os.pathsep + env.get("PYTHONPATH", "")
        subprocess.run(command, cwd=str(worktree), check=True, env=env)
        return load_json(json_path)
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(worktree)],
            cwd=str(repo_root),
            check=False,
        )


def maybe_fetch_ref(repo_root: Path, fetch_ref: str | None) -> None:
    """Fetch an optional ref from origin before comparing git versions."""
    if not fetch_ref:
        return
    subprocess.run(["git", "fetch", "origin", fetch_ref], cwd=str(repo_root), check=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # These names follow Python timeit's terminology: repeat is the number of
    # independent timing measurements, while number is the number of model
    # evaluations inside each measurement. The reported values are divided by
    # number, so they remain seconds per model evaluation. A modest number=3
    # reduces timer overhead for fast cases without making slow cases too costly.
    parser.add_argument("--repeat", type=int, default=5, help="Number of timing repeats per case.")
    parser.add_argument("--number", type=int, default=3, help="Evaluations per timing repeat.")
    parser.add_argument("--profile-limit", type=int, default=40, help="Number of cProfile rows per case.")
    parser.add_argument("--run-label", help="Label used for the current run. Defaults to the current git branch.")
    parser.add_argument("--json-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.json"))
    parser.add_argument("--md-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.md"))
    parser.add_argument("--png-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.png"))
    parser.add_argument("--html-output", type=Path, default=Path("test/benchmark_surrogate_evaluations.html"))
    parser.add_argument("--single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--script-source-kind", help=argparse.SUPPRESS)
    parser.add_argument("--script-source-ref", help=argparse.SUPPRESS)
    parser.add_argument("--script-source-path", help=argparse.SUPPRESS)
    parser.add_argument("--compare-ref", action="append", default=[], help="Git ref to benchmark in a temporary worktree.")
    parser.add_argument("--compare-label", action="append", default=[], help="Label for the matching --compare-ref.")
    parser.add_argument(
        "--fetch-ref",
        help="Fetch a ref from origin before comparison, e.g. 'pull/123/head' for a GitHub PR.",
    )
    return parser


def main() -> int:
    """Parse command-line arguments and run the requested benchmark workflow."""
    args = build_parser().parse_args()
    script_path = Path(__file__).resolve()
    repo_root = args.repo_root.resolve() if args.repo_root else script_path.parents[1]
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    if args.number < 1:
        raise SystemExit("--number must be at least 1")
    if not args.run_label:
        branch = git_value(["branch", "--show-current"], repo_root)
        args.run_label = branch if branch != "unknown" else "current"
    if not args.script_source_kind:
        args.script_source_kind = "current checkout"
    if not args.script_source_ref:
        args.script_source_ref = args.run_label
    if not args.script_source_path:
        try:
            args.script_source_path = str(script_path.relative_to(repo_root))
        except ValueError:
            args.script_source_path = str(script_path)

    if args.single_run:
        run = run_single_benchmark(args, repo_root)
        write_json(args.json_output, run)
        wrapped = {"generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(), **run}
        write_markdown(args.md_output, wrapped, args.png_output)
        write_png_table(args.png_output, wrapped)
        write_html_dashboard(args.html_output, wrapped)
        return 0

    maybe_fetch_ref(repo_root, args.fetch_ref)
    current = run_single_benchmark(args, repo_root)
    runs = [current]

    if args.compare_ref:
        with tempfile.TemporaryDirectory(prefix="gwsurrogate-bench-") as temp_dir:
            temp_root = Path(temp_dir)
            for index, ref in enumerate(args.compare_ref):
                label = args.compare_label[index] if index < len(args.compare_label) else ref
                runs.append(run_subprocess_for_ref(script_path, repo_root, ref, label, args, temp_root))

    benchmark = {
        "schema_version": 1,
        "generated_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "cases_by_model": {model: model_cases(model) for model in ENABLED_MODELS},
        "runs": runs,
    }
    write_json(args.json_output, benchmark)
    write_markdown(args.md_output, benchmark, args.png_output)
    write_png_table(args.png_output, benchmark)
    write_html_dashboard(args.html_output, benchmark)
    print(f"Wrote {args.json_output}")
    print(f"Wrote {args.md_output}")
    print(f"Wrote {args.png_output}")
    print(f"Wrote {args.html_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
