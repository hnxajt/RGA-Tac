from pathlib import Path


def ssvtp_output_root(backbone: str) -> Path:
    return Path("outputs") / f"ssvtp-{backbone}"


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
