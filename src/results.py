"""Assembling result tables.

Every `check_*` script used to build its own `pd.DataFrame({...}).explode([...])` from a
dozen parallel lists, with a different column set per study. They all now emit records in
one long format:

    <case labels...>, dataset, param, method, statistic, value

`method` is ``npe`` or ``mcmc``; `statistic` is what was measured (``median``, ``rmsd``,
...). Long format keeps the schema identical whether or not a study has ground truth, and
is what the R visualisation scripts pivot from anyway (see `visualization/load_results.R`).
"""

import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def write_long_csv(filename, records):
    """Write records to a CSV.

    Column order follows the order the keys were inserted into the first record, which is
    how the callers already build them: case labels, then `dataset`, then the measurement.
    """
    frame = pd.DataFrame.from_records(list(records))

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    frame.to_csv(filename, index=False)

    logger.info("Wrote %s rows to %s", len(frame), filename)

    return frame
