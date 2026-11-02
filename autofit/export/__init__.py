"""
Ship and check a finished results library (results-library epic, phase 4).

- ``package``: size-capped part zips with a manifest, checksums, row counts and a
  provenance README; rebuilds rewrite only the parts whose files changed;
- ``verify``: check a package is whole (checksums, CRCs, empty files, table rows,
  manifest coverage);
- ``compare``: diff two results tables — identity, exact columns and tolerance by σ;
- ``scatter_between``: the run-to-run scatter of two sets of fits, which is what a
  parity tolerance must be calibrated against;
- ``python -m autofit.export {package,verify,compare}``: the command line.
"""
from autofit.export.package import PackageException, PackageReport, index_hash, package
from autofit.export.parity import (
    ColumnParity,
    ParityReport,
    ScatterReport,
    compare,
    scatter_between,
)
from autofit.export.verify import VerifyReport, verify
