# Cognates

This tool tests whether two nouns from two Indo-European languages are related. For each pair, it computes two scores:

- **Semantic similarity (Step 3):** the word2vec cosine similarity of the two nouns' English translations.
- **Phonetic distance (Step 6):** the PanPhon weighted edit distance between the consonant skeletons of the two nouns. Before the comparison, each noun's sound changes run in reverse back to the two languages' nearest shared ancestor.

A table then lists both scores for a set of known cognates from Wiktionary, with the new pair as its last row. That lets you see where the new pair falls among established cognates.

## How it works

| Step | What happens | Main module |
|---|---|---|
| 1 | Look up each noun in Wiktionary, follow inflected forms to the lemma, and take the first English translation. | `cognates/wiktionary.py` |
| 2 | Take the lemma's IPA from Wiktionary, or from the epitran library as a fallback. | `cognates/wiktionary.py`, `cognates/ipa.py` |
| 3 | Score the two translations' head nouns with Google News word2vec. | `cognates/semantic.py` |
| 4–5 | Run each word through its language's reversed sound changes (Lexurgy rule files), back to the nearest shared ancestor. Merged sounds branch, so each word yields a set of candidate roots. Vowels are then stripped. | `cognates/reverse.py`, `cognates/tree.py`, `rules/` |
| 6 | Compute the smallest PanPhon distance over all pairs of candidate roots. | `cognates/distance.py` |
| 7 | Write a CSV table of labeled cognates, with the new pair last. | `cognates/dataset.py`, `cognates/table.py` |

The reversed sound changes come from [Index Diachronica](https://chridd.nfshost.com/diachronica/all), item 17 ("Indo-European"), with additions by the project team. Each rule file in `rules/indo-european/` names its source and marks every team addition.

## Supported languages

| Code | Language | Status |
|---|---|---|
| `en` | English | Chain drafted (7 stages to Proto-Indo-European) |
| `got` | Gothic | Chain drafted |
| `sa` | Sanskrit (Vedic) | Chain drafted (3 stages to Proto-Indo-European) |
| `nl` | Dutch | Rules not drafted yet |
| `non` | Old Norse | Rules not drafted yet |
| `ofs` | Old Frisian | Rules not drafted yet |

A pair that needs an undrafted stage stops at Step 5 with the message "reversed rules for stage … are not drafted yet." The rule files are drafts awaiting team review.

## Requirements

- **Python 3.10 to 3.13.** gensim fails to build on Python 3.14.
- **Java 8 or newer**, for the Lexurgy sound changer.
- **XeLaTeX** with the `forest` and `fontspec` packages, for the optional `--tree` switch.
- **About 10 GB of disk space** for the Wiktionary data, the word2vec vectors and the search index.

## Installation

All commands run from the project root.

### 1. System software

On macOS, Homebrew installs all three programs:

```bash
brew install python@3.12             # Python 3.12, as python3.12
brew install --cask temurin          # Java (skip if `java -version` already works)
brew install --cask mactex-no-gui    # TeX Live with XeLaTeX (only for --tree)
```

MacTeX puts its programs in `/Library/TeX/texbin`. Open a new terminal afterward, so `xelatex` appears on your `PATH`.

On Debian or Ubuntu, the matching packages are:

```bash
sudo apt install python3.12 python3.12-venv default-jre \
    texlive-xetex texlive-pictures texlive-fonts-extra
```

Check the three programs:

```bash
python3.12 --version    # Python 3.12.x
java -version           # version 8 or newer
xelatex --version       # XeTeX ... (TeX Live ...)
```

### 2. Python virtual environment

The project keeps its Python packages in a virtual environment, a private folder named `.venv` in the project root. Packages installed there stay separate from the system Python and from other projects.

Create the environment and install the project into it:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -m spacy download en_core_web_sm-3.8.0 --direct
```

- `pip install -e '.[dev]'` installs the dependencies from `pyproject.toml` (PanPhon, gensim, spaCy, epitran, PyYAML), plus pytest for the tests. Each one is pinned to an exact version, so every installation computes the same scores. The `-e` flag links the installed `cognates` command to the source folder, so code edits take effect without a reinstall.
- `spacy download en_core_web_sm-3.8.0 --direct` fetches the small English parser for Step 3, at the version that matches spaCy 3.8.

#### Optional: exact copy of the tested setup

The pins in `pyproject.toml` cover only the packages the project imports. Their own dependencies, such as numpy and scipy, install at the newest compatible version. A future release of one of them could break the install. The file `requirements.lock` fixes all 62 packages at the versions tested with the project, including the spaCy model. To install from it, use these commands in place of the four above:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.lock
.venv/bin/pip install --no-deps -e .
```

- `--no-deps` installs the `cognates` command alone and leaves the locked versions untouched.
- The lock file was generated with Python 3.12 on macOS (Apple silicon). On another platform or Python version, some pinned builds may be unavailable. In that case, use the default install above.
- After an upgrade, regenerate the file with the command in its header comment.

Activate the environment to use plain `python`, `pip` and `cognates` in the current terminal:

```bash
source .venv/bin/activate    # the prompt now starts with (.venv)
deactivate                   # return to the system Python
```

Without activation, call the programs through their full paths, e.g. `.venv/bin/cognates`.

Check the installation:

```bash
.venv/bin/python -c "import panphon, gensim, spacy, epitran; spacy.load('en_core_web_sm'); print('ok')"
```

To start over (for example, after switching Python versions), delete the folder and repeat the steps above:

```bash
rm -rf .venv
```

### 3. Lexurgy

```bash
mkdir -p tools/lexurgy
curl -sSL -o tools/lexurgy.tar https://github.com/def-gthill/lexurgy/releases/download/v1.7.7/lexurgy-1.7.7.tar
tar -xf tools/lexurgy.tar -C tools/lexurgy --strip-components=1
rm tools/lexurgy.tar
tools/lexurgy/bin/lexurgy --help
```

The tool expects Lexurgy at `tools/lexurgy/bin/lexurgy`.

### 4. Data

Wiktionary noun data, from the [kaikki.org](https://kaikki.org) dumps (about 3.9 GB; English alone is 3.4 GB):

```bash
mkdir -p data/kaikki
for L in English Gothic "Old Norse" "Old Frisian" Dutch Sanskrit; do
  N=$(echo "$L" | tr -d ' ')
  E=$(echo "$L" | sed 's/ /%20/g')
  curl -sSL -o "data/kaikki/$N.jsonl" "https://kaikki.org/dictionary/$E/kaikki.org-dictionary-$N.jsonl"
done
```

Google News word2vec vectors (about 1.6 GB):

```bash
mkdir -p data/gensim
GENSIM_DATA_DIR=$PWD/data/gensim .venv/bin/python -c \
  "import gensim.downloader as d; d.load('word2vec-google-news-300', return_path=True)"
```

The first Step 3 run converts the vectors to `data/gensim/word2vec-google-news-300.kv`, so later runs can load them quickly.

Index Diachronica, for the language tree:

```bash
mkdir -p data/diachronica
curl -sSL -o data/diachronica/all.html https://chridd.nfshost.com/diachronica/all
```

### 5. Build the index, tree and dataset

```bash
.venv/bin/cognates index           # Wiktionary noun index -> data/wiktionary.sqlite (about 30 s)
.venv/bin/cognates tree            # stage tree -> tree/indo-european.json
.venv/bin/cognates build-dataset   # labeled cognate pairs -> output/cognate_pairs.csv
```

`tree/indo-european.json` already ships with the project. Rebuild it only after changing `cognates/tree.py` or downloading a new Index Diachronica.

## Usage

The examples below call `.venv/bin/cognates`. After `source .venv/bin/activate`, a plain `cognates` works too.

### Test one pair

```bash
.venv/bin/cognates test father en पितृ sa
```

```
noun1: father -> lemma father | translation 'father' | IPA fɑːðəɹ (wiktionary-phonemic)
    reversed British-English: ...
    ...
shared ancestor: Proto-Indo-European
Step 3 semantic similarity: 1.0
Step 6 phonetic distance:   0.0 (pχtr̩ ~ pχtr̩)
```

Arguments: `noun1 lang1 noun2 lang2`, with the language codes from the table above.

- **Nouns:** type the lemma or an inflected form (*vaders* finds *vader*). Languages in other scripts accept the native script (*पितृ*, *𐍆𐌰𐌳𐌰𐍂*) or Wiktionary's romanization with its diacritics (*pitṛ́*, *fadar*, *hairtō*).
- **Output:** each word's lemma, translation and IPA, the candidate forms after each reversed stage, the shared ancestor, and both scores.
- **Lower phonetic distance means more similar.** The closest roots appear in the parentheses.

### Build the comparison table

```bash
.venv/bin/cognates table father en fadar got --langs en got sa
```

- `--langs` limits the labeled pairs to these languages. Pairs involving an undrafted language would stop at Step 5, so list only drafted ones.
- The table goes to `output/table.csv`. Pairs that stop at any step are listed, with the step and reason, in `output/skipped.csv`.
- A build for English, Gothic and Sanskrit takes about 5 minutes.

Table columns: `noun1, language1, noun2, language2, translation1, translation2, semantic_similarity, closest_root1, closest_root2, phonetic_distance`.

### Options

| Option | Commands | Effect |
|---|---|---|
| `--forgive-missing-endings` | `test`, `table` | Wiktionary gives Sanskrit nouns as bare stems, without their nominative ending. With this switch, when one word's language is marked `lemma_ending: missing` in `languages.yaml`, the other word's root may drop one final consonant at no cost. A forgiven consonant appears in parentheses, e.g. `wlf(s)`. Off by default. |
| `--output PATH` | `table` | Write the table to `PATH`. Skipped pairs go to `<name>_skipped.csv` beside it. |

### Draw a tree for a cognate pair

```bash
.venv/bin/cognates test father en पितृ sa --tree output/trees/father-pitr.tex
```

When the pair scores as cognate, this writes a LaTeX tree and compiles it into a PDF beside it (`output/trees/father-pitr.pdf`):

- **Root:** the shared consonant skeleton (e.g. \*pχtr̩), the ancestor language, and both scores.
- **Branches:** each word's derivation from its full reconstructed form at the ancestor, stage by stage, with each stage's language, down to the attested word, its IPA and its language variant (e.g. British English, Vedic Sanskrit).

A pair counts as cognate when its phonetic distance is at or below the threshold, 2.0 by default. Above it, the command prints `no tree: phonetic distance … is above the threshold …` and writes nothing.

| Option | Effect |
|---|---|
| `--tree PATH.tex` | Write the tree to `PATH.tex` and compile `PATH.pdf`. |
| `--tree-threshold D` | Largest phonetic distance that counts as cognate (default 2.0). |

Compiling needs XeLaTeX with the `forest` and `fontspec` packages, plus three fonts: Charis SIL (part of TeX Live) for IPA, and Devanagari MT and Noto Sans Gothic (both part of macOS) for Devanagari and Gothic script. On Linux, name another installed Devanagari font in the preamble of `cognates/latex_tree.py`. Without XeLaTeX, the `.tex` file is still written. The tree shows one derivation per word; where several earlier forms lead to the same later form, it takes the first in sorted order.

### Keep both scoring modes side by side

`--output` keeps a second table from overwriting the default one:

```bash
.venv/bin/cognates table father en fadar got --langs en got sa
.venv/bin/cognates table father en fadar got --langs en got sa \
    --forgive-missing-endings --output output/table_forgiving.csv
```

## Configuration

- **`languages.yaml`** lists the supported languages. Each entry names its Wiktionary file, its stage in the tree, its epitran fallback and optional settings (`prefer_tag`, `epitran_native`, `lemma_ending`). The file's header comment explains each key.
- **`rules/indo-european/<Stage>.lsc`** holds one stage's reversed sound changes in Lexurgy syntax. The matching `<Stage>.mergers.yaml` lists the sounds that branch to several ancestral sources.
- **`rules/symbols.yaml`** maps reconstruction symbols (h₂, ḱ, bʱ, …) to IPA that PanPhon can read in Step 6.

## Tests

```bash
.venv/bin/pytest -q
```

The tests need the installed data and Lexurgy, and take about a minute.

## Project layout

```
cognates/            Python package (CLI and pipeline steps)
rules/               reversed sound-change rules and the symbol table
tree/                stage tree built from Index Diachronica
tests/               test suite
languages.yaml       supported-language settings
data/                downloads and the search index (not in version control)
tools/               Lexurgy (not in version control)
output/              tables and logs (not in version control)
```

## Licenses of sources

- **Index Diachronica:** CC BY-NC-SA 3.0. The rule files derive from it, so they carry the same license and allow only noncommercial use.
- **Wiktionary data (via kaikki.org):** CC BY-SA.
- **Lexurgy:** GPL-3.0. It's downloaded, not bundled.
- **PanPhon:** MIT.
