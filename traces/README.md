# R2 Medium Office Trace Files

This directory contains gzip-compressed R2 Medium Office future-weather trace files for the zone-aggregation and tail-steering diagnostics.

Each `*_ordinal.csv.gz` file contains one city-year ordinal-control trace for Beijing, Houston, or Phoenix in 2025, 2050, or 2075 under the SSP5-8.5 weather files used in the manuscript revision. The files include aggregate thermal-comfort outputs and per-zone expected TSV, discomfort probability, warm-tail probability, cold-tail probability, and directional tail dominance.

The uncompressed combined convenience file `medium_office_control_traces.csv` is not tracked because it duplicates these nine files. To reconstruct it locally:

```sh
gzip -cd traces/*_ordinal.csv.gz > medium_office_control_traces_with_repeated_headers.csv
```

For analysis, prefer `scripts/build_zone_and_tail_diagnostics.py`, which reads the compressed city-year traces directly and handles concatenation in memory.
