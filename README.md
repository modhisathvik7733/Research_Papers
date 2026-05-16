# Research_Papers

A research lab notebook. Not a code dump — a place to do research *properly*:
read papers, form your own hypotheses, design experiments, test them, and write
down what actually happened (including when you were wrong).

This repo exists because "ask someone to implement it and hope it works" is not
research. The structure below forces the slower, correct loop.

## The loop

```
read a paper  ->  extract a question  ->  turn it into a falsifiable hypothesis
      ^                                              |
      |                                              v
  conclusion  <-  analyze results  <-  run minimal experiment (baseline first)
```

Full method in [docs/methodology.md](docs/methodology.md). Read that first.

## Layout

| Folder           | What goes here                                                        |
|------------------|-----------------------------------------------------------------------|
| `papers/`        | One note per paper you actually read. Use `papers/TEMPLATE.md`.        |
| `ideas/`         | Raw ideas, scored and triaged before they become experiments.         |
| `experiments/`   | One folder per experiment. Hypothesis → config → log → result.         |
| `lab_notebook/`  | Dated daily entries. What you did, what you learned, what's next.      |
| `src/researchkit/` | Shared, reused code (seeding, logging, metrics). Not experiment code. |
| `docs/`          | The methodology and any longer write-ups.                             |

## Rules that make this work

1. **No experiment without a written hypothesis first.** If you can't state what
   would prove you wrong, you're not testing anything.
2. **Baseline before variant.** Reproduce the known number before you change it.
3. **One variable per experiment.** Otherwise you can't attribute the result.
4. **Negative results are committed too.** A killed idea is real progress.
5. **Every experiment is reproducible**: seed + config + environment captured.

## Setup

```bash
uv sync                      # create env from pyproject.toml
uv run python -c "import researchkit; print(researchkit.__version__)"
```

## Daily start

```bash
cp lab_notebook/_TEMPLATE.md lab_notebook/$(date +%F).md
$EDITOR lab_notebook/$(date +%F).md
```
