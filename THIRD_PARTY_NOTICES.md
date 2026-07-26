# Third-party notices

The repository license covers only original project materials. The following
external software and reference data retain their own terms and attribution.

## NASA General Mission Analysis Tool

Files under `data/reference/gmat_r2026a` were copied from, or recorded for
comparison with, GMAT R2026a. The included JGM2 gravity coefficient file and
exponential-atmosphere table retain their original notices and are provided
only to reproduce the recorded validation workflow.

GMAT is distributed by NASA under the Apache License 2.0:

- https://github.com/nasa/GMAT
- `licenses/Apache-2.0.txt`

No endorsement by NASA or the GMAT project is implied.

## International Earth Rotation and Reference Systems Service

`data/reference/gmat_r2026a/eopc04_08.62-now` is an IERS Earth-orientation
parameter record preserved for deterministic comparison. The file is not
owned by this project. Users should consult the official IERS data pages for
current products, attribution, and use information:

- https://www.iers.org/iers/en/dataproducts/data

The frozen file must not be interpreted as a current operational EOP source.

## CelesTrak TLE snapshot

The ISS TLE under `data/tle` was retrieved from CelesTrak on
2026-07-18T19:32:54Z using NORAD catalog number 25544. The source query and
retrieval metadata are preserved in
`data/tle/iss_25544_2026-07-18_celestrak_metadata.json`.

- https://celestrak.org/

TLE data are time-specific inputs. A newer TLE will not reproduce the frozen
Paper 1 case.

## Map background

`data/maps/world_texture.png` is a project-created neutral visualization
background. It does not participate in orbit propagation, coordinate
transformation, station-access calculations, or validation acceptance.

## Python dependencies

NumPy, SciPy, Matplotlib, Astropy, and the Python `sgp4` package are installed
separately and retain their respective upstream licenses. They are not
redistributed as source or binary packages in this repository.
