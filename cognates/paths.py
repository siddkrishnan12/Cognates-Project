"""Project-relative locations of data, tools and outputs."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
KAIKKI = DATA / "kaikki"
GENSIM = DATA / "gensim"
LEXURGY = ROOT / "tools" / "lexurgy" / "bin" / "lexurgy"
RULES = ROOT / "rules"
TREE = ROOT / "tree"
OUTPUT = ROOT / "output"
