"""Fetch the two public inputs into data/.  Run:  python -m pipeline.download

1. October 2026 Delhi bus e-ticketing data (hackathon release, ~66 MB zip)
2. Delhi bus static GTFS (DTC/DIMTS, Mobility Database mirror mdb-3139, ~19 MB zip).
   If you have the newer GTFS from delhi.transportstack.in, unzip it into data/gtfs/ instead.
"""
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

TICKETS = "https://pub-091bd9e794bb4877b10b29e6df80f065.r2.dev/2026-10/October-Delhi.csv.zip"
GTFS = "https://files.mobilitydatabase.org/mdb-3139/mdb-3139-202605111616/mdb-3139-202605111616.zip"


def fetch(url, dest: Path):
    if dest.exists():
        print("have", dest.name)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)


def main():
    raw = config.RAW_TICKETS.parent
    z = raw / "October-Delhi.csv.zip"
    fetch(TICKETS, z)
    if not config.RAW_TICKETS.exists():
        with zipfile.ZipFile(z) as f:
            f.extract("October-Delhi.csv", raw)
    g = config.GTFS_DIR / "delhi-gtfs.zip"
    fetch(GTFS, g)
    if not (config.GTFS_DIR / "stop_times.txt").exists():
        with zipfile.ZipFile(g) as f:
            for n in f.namelist():
                if n.endswith(".txt") and "/" not in n:
                    f.extract(n, config.GTFS_DIR)
    print("ready")


if __name__ == "__main__":
    main()
