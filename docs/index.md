# t-boost

t-boost is a machine learning algorithm that uses gradient boosting on oblivious decision
trees. Its fitted model is *exactly* a set of rating tables: a Tabulating Boosting Machine
(TBM). It is available as an open source library.

Gradient-boosted trees are usually more accurate than a GLM, but they are hard to read, review
or deploy in systems built around rating tables. t-boost aims to get boosting-level accuracy
in a model that *is* a set of main-effect and interaction tables, with no approximation and no
surrogate model.

- [Installation](installation/index.md)
- [Quick start](quickstart.md)
