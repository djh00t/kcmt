import yaml
from pathlib import Path
from states import AssimilationState
from analysis.python_ast import run_ast_analysis
from analysis.dependency_graph import run_dependency_graph
from analysis.coverage_map import run_coverage_map

REPORT_DIR = Path("keystone/tools/assimilation/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = Path(".keystone/assimilation.yaml")

def load_config():
    if not CONFIG_PATH.exists():
        raise RuntimeError("Missing .keystone/assimilation.yaml")
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)["keystone"]

def write_report(name, data):
    path = REPORT_DIR / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(data, f)

def main():
    cfg = load_config()
    state = AssimilationState(cfg["state"])

    report = {
        "project": cfg["project_id"],
        "state": state.value,
        "signals": {},
        "enforcement": {},
    }

    # --- AST / Static Analysis (always allowed) ---
    if cfg["analysis"]["enabled"]:
        report["signals"]["ast"] = run_ast_analysis()
        report["signals"]["dependencies"] = run_dependency_graph()
        report["signals"]["coverage"] = run_coverage_map()

    # --- Enforcement gates (forward-only) ---
    if state in {AssimilationState.GOVERNED,
                 AssimilationState.CANONICAL,
                 AssimilationState.IPBANKED}:
        enforce_governed(cfg, report)

    write_report("assimilation_report", report)

def enforce_governed(cfg, report):
    enforcement = report["enforcement"]

    # Enforce make check
    if cfg["enforcement"]["require_make_check"]:
        result = os.system("make check")
        if result != 0:
            raise RuntimeError("make check failed")

    # Enforce coverage
    min_cov = cfg["enforcement"]["min_coverage"]
    coverage = report["signals"]["coverage"]["total"]
    if coverage < min_cov:
        raise RuntimeError(
            f"Coverage {coverage}% < required {min_cov}%"
        )

    enforcement["status"] = "passed"

if __name__ == "__main__":
    main()