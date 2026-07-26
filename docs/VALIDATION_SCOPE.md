# Validation scope and limitations

## Models in the Paper 1 comparison

The central model set is analytical two-body, numerical two-body, numerical
J2, numerical J2 plus simplified exponential drag, and fixed-TLE SGP4.
Higher-order gravity work is retained as supporting validation but is not a
central Paper 1 claim.

## Controlled cases

LEO400 and SSO700 use defined inertial benchmark initial conditions at
2026-01-01T00:00:00Z. The ISS case uses NORAD 25544 with a TLE epoch of
2026-07-18T02:08:01.938048Z.

## Frames and Earth orientation

Controlled Cartesian cases use the registered Earth-centered inertial
benchmark frame. SGP4 returns TEME states. The implementation transforms those
states into GCRS using Astropy for model comparison and into ITRS for Earth
fixed ground-track and station-access calculations.

The closed validation uses the saved Earth-orientation record. Polar motion,
precession-nutation, sidereal rotation, and time-scale handling are therefore
part of the recorded comparison and not silently downloaded at run time.

## Simplified drag

The drag model is a deterministic exponential-atmosphere sensitivity model.
It uses atmospheric-relative velocity and converts between kilometre-based
state units and SI drag-law inputs. It is not a space-weather-driven density
forecast and does not claim operational decay prediction.

GMAT validation establishes agreement of the drag acceleration calculation
for four controlled scenarios and 25 checks. It does not independently
validate long-arc atmospheric truth.

## SGP4 comparison

The ISS experiment compares numerical models with a fixed-TLE SGP4 trajectory.
It is not an ephemeris-truth or tracking-data validation. Separation values
measure model divergence from the selected TLE and epoch.

## Ground passes

Station visibility is geometric. The Bremen station uses latitude 53.0793
degrees north, longitude 8.8017 degrees east, altitude 12 m, and a 10-degree
elevation mask. Events use a 20 s coarse grid, PCHIP interpolation, Brent root
finding, and a 600 s pass-matching window. Refraction, terrain masking,
antenna constraints, and link budget are excluded.

## Appropriate use

The project is suitable for reproducible research, teaching, sensitivity
studies, and software verification. It is not flight-qualified and must not be
used alone for operational orbit determination, collision-risk decisions, or
spacecraft control.
