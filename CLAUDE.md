# t-boost: design principles

Read `CONTRIBUTING.md` before changing anything: it is the operating manual (repository map,
environment, the gates that define "done", code and writing rules). This file holds the aims
every change is judged against.

## The aim

Gradient boosting machines fit noise along with the signal. t-boost strips that noise out, so the
model both **predicts better** and **is easier to explain**. Every stage after the boost serves
this aim: pruning drops tables whose evidence is noise, banding merges bins whose differences are
noise, and graduation smooths noise out of the tables. Judge a change by held-out deviance and by
how readable the resulting tables are.

## Principles

- **The model is always a set of rating tables.** Every fit deploys tables, whatever the settings
  (`prune=False` deploys the full, unpruned bank). Never deploy a tree ensemble, a surrogate or an
  approximation in their place.
- **Performant.** Heavy or per-row work belongs in the Rust core; Python only orchestrates. Measure
  the speed cost of a change, not just its accuracy.
- **Robust.** Behave sensibly across objectives, data sizes, weights and exposure, categorical
  features, grouped (panel) data and monotone constraints, without special cases for each.
- **Avoid over-complication.** Prefer the simplest mechanism that works. A new branch, knob or
  stage has to earn its place with measured evidence. Propose removing ones that don't.
- **Prefer parameters that don't need tuning.** Defaults should be right across datasets. Where a
  choice depends on the data, derive it from the data (for example, from held-out evidence) rather
  than adding a parameter the user has to tune.
- **No behaviour completely switched off by a setting.** Every stage (selection/pruning, banding,
  the guard, budgets, graduation) runs for every supported configuration. When a setting imposes a
  constraint a stage could break, change the stage so it respects the constraint; don't skip the
  stage. For example, every stage must still run under monotone constraints, and the deployed bank
  must still be monotone. If a stage genuinely can't honour a setting yet, fail loudly with an
  error that names the conflict, never skip it silently, and treat it as a gap to close.
