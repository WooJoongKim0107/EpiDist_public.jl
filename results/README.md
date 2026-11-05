# Archived figure data

These CSVs use the stored manuscript inputs. Figures 2, 3, 6, and 7
retain all 555,800 coordinates. The example in the main README computes
a separate 14,000-point grid.

- Figures 2, 3, 6, and 7 use the AO timing cache, LF/CO/AO records, and
  saved channel classifications. Figure 6 includes the crowding-in mask.
- Figure 1 uses selected saved equilibrium records to reconstruct the time
  series and adds the corresponding no-distancing trajectory.
- Figure 4 contains the CO and LF investment-response curves from the
  saved deviator records. Its channel columns are calculated from those
  CO records using the integration tolerances in the manuscript plotting
  code. LF channel cells are blank because the LF insets show only investment.
- Figure 5 evaluates the time-series sensitivities from the four saved
  deviator records used in the manuscript.
- `boundaries.csv` contains the LF/CO phase curves and saved zone paths.
