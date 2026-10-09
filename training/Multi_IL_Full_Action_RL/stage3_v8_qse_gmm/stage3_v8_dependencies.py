"""Fail-closed hashes for inherited project code, runtime configs and V8 sources."""
import hashlib,json,sys,subprocess
from pathlib import Path
import stage3_v8_paths as paths
COMMIT="891ff330f7f3348065597c3ee76a150236705a65"
LOCK=paths.HERE/"stage3_v8_dependency_lock.json"
def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_dependencies(lock_path=LOCK,root=None):
    root=Path(root or paths.ROOT)
    lock=json.loads(Path(lock_path).read_text())
    failures=[]
    for relative,expected in lock["files"].items():
        f=root/relative
        if not f.is_file() or digest(f)!=expected:
            failures.append(relative)
    if failures: raise RuntimeError("V8 dependency hash mismatch: "+", ".join(failures))
    for absolute,expected in lock.get("external_files",{}).items():
        if not Path(absolute).is_file() or digest(absolute)!=expected:
            raise RuntimeError("External runtime source mismatch: "+absolute)
    import importlib.metadata
    for package,expected in lock.get("package_versions",{}).items():
        if importlib.metadata.version(package)!=expected:
            raise RuntimeError("Runtime package version mismatch: "+package)
    expected_schedule=(paths.PINNED/"stage3_v7_schedule.py").resolve()
    loaded=sys.modules.get("stage3_v7_schedule")
    if loaded is not None and Path(loaded.__file__).resolve()!=expected_schedule:
        raise RuntimeError("Unpinned V7 schedule already imported")
    import importlib.util
    if Path(importlib.util.find_spec("stage3_v7_schedule").origin).resolve()!=expected_schedule:
        raise RuntimeError("V7 schedule import resolution is not pinned")
    return dict(status="PASS",locked_files=len(lock["files"]),reference_commit=lock["reference_commit"],
                schedule=str(expected_schedule),lock_sha256=digest(lock_path))
def audit_loaded_dependencies():
    lock=json.loads(LOCK.read_text());missing=[]
    for name,module in list(sys.modules.items()):
        f=getattr(module,"__file__",None)
        if not f: continue
        try: relative=str(Path(f).resolve().relative_to(paths.ROOT))
        except ValueError: continue
        if relative.endswith(".py") and relative not in lock["files"]:
            # Diagnostic tests are separately recorded, never part of the formal entry.
            if "/testing/" not in relative: missing.append([name,relative])
    if missing: raise RuntimeError("Loaded unlocked project modules: "+repr(missing))
    return dict(status="PASS",unlocked_project_modules=[])
