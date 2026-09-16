"""Load and clean French Motor TPL frequency and severity files."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

FREQ_NAME = "freMTPL2freq.csv"
SEV_NAME = "freMTPL2sev.csv"

USER_DOWNLOADS = Path.home() / "Downloads"

# Hugging Face mirror of the CASdatasets CSVs (not committed to this repo).
_HF = "https://huggingface.co/datasets/mabilton/fremtpl2/resolve/main"
FREQ_URL = f"{_HF}/{FREQ_NAME}"
SEV_URL = f"{_HF}/{SEV_NAME}"
_UA = "Frequency-Severity-GLM-Pricing/1.0"


def project_paths(root: Path | None = None) -> dict[str, Path]:
    root = root or Path(__file__).resolve().parents[1]
    raw = root / "data" / "raw"
    processed = root / "data" / "processed"
    outputs = root / "outputs"
    raw.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "raw": raw,
        "processed": processed,
        "outputs": outputs,
        "freq": raw / FREQ_NAME,
        "sev": raw / SEV_NAME,
    }


def _copy_if_needed(src: Path, dest: Path) -> None:
    if dest.exists():
        return
    if not src.exists():
        return
    dest.write_bytes(src.read_bytes())


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _ensure_file(dest: Path, url: str, label: str) -> None:
    if dest.exists():
        return
    print(f"Downloading {label} from Hugging Face …", flush=True)
    try:
        _download(url, dest)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Missing {dest.name}. Place it in {dest.parent} or {USER_DOWNLOADS}. "
            f"Tried Hugging Face mirror and failed: {exc}"
        ) from exc
    print(f"Wrote {dest}", flush=True)


def ensure_raw_files(root: Path | None = None) -> dict[str, Path]:
    paths = project_paths(root)
    _copy_if_needed(USER_DOWNLOADS / FREQ_NAME, paths["freq"])
    _copy_if_needed(USER_DOWNLOADS / SEV_NAME, paths["sev"])
    _ensure_file(paths["freq"], FREQ_URL, FREQ_NAME)
    _ensure_file(paths["sev"], SEV_URL, SEV_NAME)
    return paths


def load_frequency(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["IDpol"] = df["IDpol"].astype("int64")
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4).astype(int)
    df["Exposure"] = df["Exposure"].clip(upper=1.0)
    df = df.drop_duplicates("IDpol", keep="first")
    return df


def load_severity(path: Path) -> pd.DataFrame:
    sev = pd.read_csv(path)
    sev["IDpol"] = sev["IDpol"].astype("int64")
    sev = sev.loc[sev["ClaimAmount"] > 0].copy()
    return sev
