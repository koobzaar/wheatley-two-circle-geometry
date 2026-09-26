"""The symbolic checks behind the written proofs of the technical note (sim/scripts/technote_proof_checks.py)."""
import importlib.util
import pathlib


def test_written_proofs_symbolic_checks():
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "technote_proof_checks.py"
    spec = importlib.util.spec_from_file_location("technote_proof_checks", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.run() == []
