# Scripts

- `predict_compartment.sh`: generic shell wrapper for checkpoint prediction plus 100 kb aggregation.
- `predict_ranges.py`: stream range-level model predictions to a TSV file.
- `aggregate_predictions.py`: aggregate streamed predictions to fixed-width bins and compute metrics.

Run scripts from the `CrossCompartment/` directory so Python imports resolve against the local `src/` package.
