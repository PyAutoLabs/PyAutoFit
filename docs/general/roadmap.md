(roadmap)=

# Road Map

**PyAutoFit** is in active development and the road-map of features currently planned in the short and long term are
listed and described below:

**JAX:**

Supported for autodiff is nearly implemented, including JAX fits.

**Non-Linear Searches:**

We are always striving to add new non-linear searches to PyAutoFit\*. In the short term, we aim to provide a wrapper to the many method available in the `scipy.optimize` library with support for outputting results to hard-disk.

If you would like to see a non-linear search implemented in **PyAutoFit** please [raise an issue on GitHub](https://github.com/PyAutoLabs/PyAutoFit/issues)!

**Approximate Bayesian Computation**

Approximate Bayesian Computational (ABC) allows for one to infer parameter values for likelihood functions that are
intractable, by simulating many datasets and extracting from them a summary statistic that is compared to the
observed dataset.

ABC in **PyAutoFit** will be closely tied to the Database tools, ensuring that the simulation, fitting and extraction
of summary statistics can be efficiently scaled up to extremely large datasets.
