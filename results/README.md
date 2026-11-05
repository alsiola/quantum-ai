# results/

Raw run outputs go in `results/<experiment>/<UTC timestamp>/` and are gitignored.
Small, curated summaries worth keeping (tables, final metrics, plots used in
reports) go in `results/summaries/` and are committed. Every saved result
carries the metadata from `qmem.records.run_metadata()` (commit, dirty flag,
environment, config, seed).
