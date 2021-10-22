# DepMap Drug Similarity Exploration

## PRISM Paper (2016) Summary

Resources:
* PRISM paper (2016): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5508574
* PRISM original protocol: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5508574/bin/NIHMS744739-supplement-2.pdf
* PRISM method summary: https://www.theprismlab.org/the-prism-assay
* PRISM code: https://github.com/broadinstitute/prism_data_processing

Summary of method:
* Use viral vectors to barcode each cell line
* Pool cell lines together (25 cell lines per pool, grouped by doubling times as of 2020); don't worry about effects they might exert on one another (!)
* Plate each pool in a 384-well assay plate with 1,250 cells from that pool per well (50 cells per cell line)
* Each plate of wells is therefore per-cell-line-pool (many drugs per plate)
* Treat wells with compounds (1:1 well mapping from compound plate to cell line pool plate), some of which have known MOAs*
* Lyse cells and amplify barcode sequences
* Mix in Luminex microspheres of different colors that hybridize as one color per barcode
* Measure relative amounts of each barcode remaining using Luminex scanner (fluorescence)
* Post-processing/filtering of data**

Definitions:
* Compound plate: a 384-well plate containing a mix of many compounds and some controls
* Detection plate: three (replicate) 384-well plates with cell line pools treated with compounds
* HTS vs. MTS screens: HTS is pin-transfer directly from a compound plate

*Example: Known MOA BRAF inhibitor AZ628 is included; we expect cell lines with BRAFV600E mutation to be more sensitive (this is shown)
**e.g. ignore cell lines if they don't react very differently to negative vs. positive controls:
 * Negative controls: treated only with the vehicle solution (e.g. saline)
 * Positive controls: treated with a potent killer-of-everything

## PRISM Non-Oncology Paper (2020) Summary

Resources:
* PRISM non-oncology drugs paper (2020): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7328899

Summary:
* Use PRISM to test 4,518 drugs (3,466 non-oncology-related) against 578 cell lines
* Screen in two stages:
 * Primary: all drugs screened, with three replicate wells per drug-pool
 * Secondary: 1,448 of those drugs were chosen for dose-response testing at 8 dose levels
