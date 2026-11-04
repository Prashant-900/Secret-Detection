# Local verification

The reporting bundle was tested with the project's actual suite/assessment schemas and a small, explicitly fictional fixture. These checks validate the reporting software; they are not measurements of the remotely trained model.

- 22 automated tests passed, including a complete report with actual PNG/SVG generation and embedded offline HTML.
- Hand-calculated tied-score AP/AUC, empty/one-class data, unknown labels, threshold inclusivity, calibration bins, and seed sample standard deviation were checked.
- Integration checks cover real/synthetic separation, frozen BLOCK diagnostics, ensemble-first seed summaries, partial reports, stale hashes/metrics/thresholds, and relocated documentation links.
- Raw fixture credential text was absent from exported HTML, JSON, Markdown, CSV and SVG.
- Sixteen plotting cases were rendered and visually inspected, including long labels, missing data, and negative augmentation effects.
- A complete local fictional report produced 46 PNG/SVG figure pairs. Your plot count depends on the saved remote artifacts.

Run the same suite on the remote machine after installing the reporting requirements:

```bash
.venv-train/bin/python -m unittest discover -s reporting_bundle/tests -v
```

Remote GPU training and test scores are read from your artifacts when you run the report; they have not been inferred from the local verification fixture.
