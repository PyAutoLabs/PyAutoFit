The `non_linear` folder contains configuration files which customize the default behaviour of non-linear searches in
**PyAutoFit**.

The default settings of each individual search (e.g. `Nautilus`, `Emcee`, `LBFGS`) are not set by configuration files.
They are the default values of the arguments of each search class's constructor, and are changed by passing those
arguments when the search is created.

# Files

- `GridSearch.yaml`: Settings default behaviour of a parallelized grid search of non-linear searches (e.g. the
  number of cores and the default step size).
