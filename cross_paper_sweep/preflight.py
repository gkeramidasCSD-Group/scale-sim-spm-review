#!/usr/bin/env python3
"""Sanity checks to run before the cross-paper sweep.

Adapted from sim-opt:benchmark/preflight.py for 3 named paper worktrees
(cosma, onsram, smm) instead of 2 compared versions. Verifies each
worktree has its entry point, each venv's python actually imports
scalesim/numpy from under THAT worktree (not a stale system/user-site
install -- a confirmed real failure mode on this machine, see
_import_probe.py's own docstring), the shared cosma/_exported model cache
resolves and is non-empty from every worktree, and (cosma only) gurobipy
is importable, since cosma/run_experiments.py defaults to --solver
gurobi. Exits non-zero with every problem found at once, rather than
letting the first one surface partway through a multi-hour unattended
sweep.
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

ENTRY_POINTS = {
    "cosma": "cosma/run_experiments.py",
    "onsram": "onsram/run_onsram.py",
    "smm": "smm/run_smm.py",
}

_PROBE_SCRIPT = os.path.join(HERE, "_import_probe.py")


def check_files(name, root):
    problems = []
    entry = os.path.join(root, ENTRY_POINTS[name])
    if not os.path.isfile(entry):
        problems.append(f"[{name}] missing entry point: {entry}")

    exported = os.path.join(root, "cosma", "_exported")
    real = os.path.realpath(exported)
    if not os.path.isdir(real):
        problems.append(f"[{name}] cosma/_exported does not resolve to a directory: "
                         f"{exported} -> {real}")
    elif not os.listdir(real):
        problems.append(f"[{name}] cosma/_exported resolves but is empty: {real}")
    return problems


def check_python(name, venv_python, root):
    """Same cwd-trick as the sim-opt precedent: run the probe as a plain
    script from HERE (which has no nested scalesim/ of its own) so
    resolution goes through the venv's real site-packages/editable-install
    path, then confirm the resolved scalesim lives under THIS worktree's
    root, not some other one."""
    problems = []
    numpy_version = None
    try:
        out = subprocess.run([venv_python, _PROBE_SCRIPT], cwd=HERE,
                              capture_output=True, text=True, timeout=60)
        if out.returncode != 0:
            problems.append(f"[{name}] '{venv_python}' failed to import scalesim/numpy:\n{out.stderr}")
        else:
            lines = out.stdout.strip().splitlines()
            if len(lines) < 2:
                problems.append(f"[{name}] '{venv_python}' probe produced unexpected output: {out.stdout!r}")
            else:
                scalesim_file, numpy_version = lines[0], lines[1]
                resolved_dir = os.path.realpath(os.path.dirname(os.path.dirname(scalesim_file)))
                expected_dir = os.path.realpath(root)
                if resolved_dir != expected_dir:
                    problems.append(
                        f"[{name}] '{venv_python}' imports scalesim from {scalesim_file}, "
                        f"which is NOT under this worktree's root ({root}) -- this venv is "
                        f"not correctly isolated for this worktree (stale system-wide or "
                        f"user-site install shadowing it?). Fix before running the sweep, or "
                        f"every run under '{name}' will silently test the wrong code.")
    except FileNotFoundError:
        problems.append(f"[{name}] venv python not found: {venv_python}")
    except subprocess.TimeoutExpired:
        problems.append(f"[{name}] import probe timed out")
    return problems, numpy_version


def check_gurobi(venv_python):
    """cosma/run_experiments.py defaults to --solver gurobi -- confirm the
    license-bearing python can actually import it before the sweep trusts
    that default, rather than discovering a silent per-row failure later."""
    out = subprocess.run([venv_python, "-c", "import gurobipy; print(gurobipy.__version__)"],
                          capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        return [f"[cosma] gurobipy not importable in {venv_python} -- "
                f"manifest rows will need --solver cbc instead of the default:\n{out.stderr}"]
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktrees-json", default=os.path.join(HERE, "worktrees.json"))
    ap.add_argument("--skip-papers", default="",
                     help="comma-separated paper names (cosma/onsram/smm) this machine "
                          "intentionally never runs (e.g. a machine only ever invoked "
                          "with --only paper=smm doesn't need cosma's own venv "
                          "functional, even though cosma's WORKTREE/cache must still "
                          "exist for model-path resolution -- see anchors.py's "
                          "shared_model_path(), which always resolves through the "
                          "'cosma' entry regardless of which paper is asking). Still "
                          "runs check_files() for skipped papers (cheap, and the cache "
                          "symlink matters regardless of --only); only skips the "
                          "venv-import and gurobi checks for them.")
    args = ap.parse_args()
    skip = {p.strip() for p in args.skip_papers.split(",") if p.strip()}

    with open(args.worktrees_json) as f:
        worktrees = json.load(f)

    all_problems = []
    numpy_versions = {}
    skipped = []

    for name, info in worktrees.items():
        all_problems += check_files(name, info["root"])
        if name in skip:
            skipped.append(name)
            continue
        problems, numpy_version = check_python(name, info["venv_python"], info["root"])
        all_problems += problems
        numpy_versions[name] = numpy_version

    if "cosma" in worktrees and "cosma" not in skip:
        all_problems += check_gurobi(worktrees["cosma"]["venv_python"])

    versions_seen = {v for v in numpy_versions.values() if v}
    if len(versions_seen) > 1:
        print(f"WARNING: numpy version mismatch across worktrees: {numpy_versions}. "
              f"Not blocking, but worth knowing before comparing timing across papers.")

    if skipped:
        print(f"Skipped venv/gurobi checks for: {skipped} (--skip-papers)")

    print()
    if all_problems:
        print(f"PREFLIGHT: {len(all_problems)} problem(s) found:")
        for p in all_problems:
            print(f"  - {p}")
        sys.exit(1)

    print("PREFLIGHT: all checks passed.")
    print(f"  numpy versions: {numpy_versions}")
    sys.exit(0)


if __name__ == "__main__":
    main()
