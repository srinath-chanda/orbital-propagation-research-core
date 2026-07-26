# Frozen GMAT R2026a Earth-orientation evidence

`eopc04_08.62-now` is the exact EOP file in the official NASA GMAT `R2026a`
source tag. Research Core 1C.1 checksum-gates this file and follows the tagged
GMAT rotation order, interpolation rules, sign convention, and endpoint
clamping behavior.

The file ends at 2026-09-14. The saved October 2026 case is therefore evaluated
with that last row, matching GMAT R2026a. This is deliberate. Substituting a
newer final EOP series would answer a different scientific question and would
not reproduce the saved GMAT evidence.

See `PROVENANCE.json` for the source tag, commit, URLs, checksum, coverage, and
the GMAT source files used for implementation conformance.

Some Windows GMAT distributions store the identical text with CRLF rather than
LF line endings. Research Core 1C.3 accepts that representation only when
normalizing CRLF to LF produces the exact frozen SHA-256 above; numeric or
other textual changes still fail verification.
