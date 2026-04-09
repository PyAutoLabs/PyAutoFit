.. _search:

Search
======

This cookbook provides an overview of the non-linear searches available in **PyAutoFit**, and how to use them.

**Contents:**

It first covers standard options available for all non-linear searches:

- **Example Fit**: A simple example of a non-linear search to remind us how it works.
- **Output To Hard-Disk**: Output results to hard-disk so they can be inspected and used to restart a crashed search.
- **Output Customization**: Customize the output of a non-linear search to hard-disk.
- **Unique Identifier**: Ensure results are output in unique folders, so tthey do not overwrite each other.
- **Iterations Per Update**: Control how often non-linear searches output results to hard-disk.
- **Parallelization**: Use parallel processing to speed up the sampling of parameter space.
- **Plots**: Perform non-linear search specific visualization using their in-built visualization tools.
- **Start Point**: Manually specify the start point of a non-linear search, or sample a specific region of parameter space.

It then provides example code for using every search:

- **Emcee (MCMC)**: The Emcee ensemble sampler MCMC.
- **Zeus (MCMC)**: The Zeus ensemble sampler MCMC.
- **DynestyDynamic (Nested Sampling)**: The Dynesty dynamic nested sampler.
- **DynestyStatic (Nested Sampling)**: The Dynesty static nested sampler.
- **LBFGS**: The L-BFGS scipy optimization.

Example Fit
-----------

An example of how to use a ``search`` to fit a model to data is given in other example scripts, but is shown below
for completeness.

.. code-block:: python

    dataset_path = path.join("dataset", "example_1d", "gaussian_x1")
    data = af.util.numpy_array_from_json(file_path=path.join(dataset_path, "data.json"))
    noise_map = af.util.numpy_array_from_json(
        file_path=path.join(dataset_path, "noise_map.json")
    )

    model = af.Model(af.ex.Gaussian)

    analysis = af.ex.Analysis(data=data, noise_map=noise_map)

It is this line, where the command ``af.Emcee()`` can be swapped out for the examples provided throughout this
cookbook to use different non-linear searches.

.. code-block:: python

    search = af.Emcee()

    result = search.fit(model=model, analysis=analysis)

Output To Hard-Disk
-------------------

By default, a non-linear search does not output its results to hard-disk and its results can only be inspected
in a Jupyter Notebook or Python script via the ``result`` object.

However, the results of any non-linear search can be output to hard-disk by passing the ``name`` and / or ``path_prefix``
attributes, which are used to name files and output the results to a folder on your hard-disk.

The benefits of doing this include:

- Inspecting results via folders on your computer is more efficient than using a Jupyter Notebook for multiple datasets.
- Results are output on-the-fly, making it possible to check that a fit is progressing as expected mid way through.
- Additional information about a fit (e.g. visualization) can be output.
- Unfinished runs can be resumed from where they left off if they are terminated.
- On high performance super computers results often must be output in this way.

The code below shows how to enable outputting of results to hard-disk:

.. code-block:: python

    search = af.Emcee(
        path_prefix=path.join("folder_0", "folder_1"),
        name="example_mcmc"
    )


These outputs are fully described in the scientific workflow example.

Output Customization
--------------------

For large model fitting problems outputs may use up a lot of hard-disk space, therefore full customization of the 
outputs is supported. 

This is controlled by the ``output.yaml`` config file found in the ``config`` folder of the workspace. This file contains
a full description of all customization options.

A few examples of the options available include:

- Control over every file which is output to the ``files`` folder (e.g. ``model.json``, ``samples.csv``, etc.).

- For the ``samples.csv`` file, all samples with a weight below a certain value can be automatically removed.

- Customization of the ``samples_summary.json`` file, which summarises the results of the model-fit  (e.g. the maximum
  log likelihood model, the median PDF model and 3 sigma error). These results are computed using the full set of
  samples, ensuring samples removal via a weight cut does not impact the results.

In many use cases, the ``samples.csv`` takes up the significant majority of the hard-disk space, which for large-scale
model-fitting problems can exceed gigabytes and be prohibitive to the analysis. 

Careful customization of the ``output.yaml`` file enables a workflow where the ``samples.csv`` file is never output, 
but all important information is output in the ``samples_summary.json`` file using the full samples to compute all 
results to high numerical accuracy.

Unique Identifier
-----------------

Results are output to a folder which is a collection of random characters, which is the 'unique_identifier' of
the model-fit. This identifier is generated based on the model fitted and search used, such that an identical
combination of model and search generates the same identifier.

This ensures that rerunning an identical fit will use the existing results to resume the model-fit. In contrast, if
you change the model or search, a new unique identifier will be generated, ensuring that the model-fit results are
output into a separate folder.

A ``unique_tag`` can be input into a search, which customizes the unique identifier based on the string you provide.
For example, if you are performing many fits to different datasets, using an identical model and search, you may
wish to provide a unique tag for each dataset such that the model-fit results are output into a different folder.

.. code-block:: python

    search = af.Emcee(unique_tag="example_tag")

Iterations Per Update
---------------------

If results are output to hard-disk, this occurs every ``iterations_per_full_update`` number of iterations.

For certain problems, you may want this value to be low, to inspect the results of the model-fit on a regular basis.
This is especially true if the time it takes for your non-linear search to perform an iteration by evaluating the 
log likelihood is long (e.g. > 1s) and your model-fit often goes to incorrect solutions that you want to monitor.

For other problems, you may want to increase this value, to avoid spending lots of time outputting the results to
hard-disk. This is especially true if the time it takes for your non-linear search to perform an iteration by
evaluating the log likelihood is fast (e.g. < 0.1s) and you are confident your model-fit will find the global
maximum solution given enough iterations.

.. code-block:: python

    search = af.Emcee(iterations_per_full_update=1000)

Parallelization
---------------

Many searches support parallelization using the Python ````multiprocessing```` module. 

This distributes the non-linear search analysis over multiple CPU's, speeding up the run-time roughly by the number 
of CPUs used.

To enable parallelization, input a ``number_of_cores`` greater than 1. You should aim not to exceed the number of
physical cores in your computer, as using more cores than exist may actually slow down the non-linear search.

.. code-block:: python

    search = af.Emcee(number_of_cores=4)

Plots
-----

Every non-linear search supported by **PyAutoFit** has a dedicated ``plotter`` class that allows the results of the
model-fit to be plotted and inspected.

This uses that search's in-built visualization libraries, which are fully described in the ``plot`` package of the
workspace.

For example, ``Emcee`` has a corresponding ``EmceePlotter``, which is used as follows.

Checkout the ``plot`` package for a complete description of the plots that can be made for a given search.

.. code-block:: python

    samples = result.samples

    plotter = aplt.MCMCPlotter(samples=samples)

    plotter.corner(
        bins=20,
        range=None,
        color="k",
        hist_bin_factor=1,
        smooth=None,
        smooth1d=None,
        label_kwargs=None,
        titles=None,
        show_titles=False,
        title_fmt=".2f",
        title_kwargs=None,
        truths=None,
        truth_color="#4682b4",
        scale_hist=False,
        quantiles=None,
        verbose=False,
        fig=None,
        max_n_ticks=5,
        top_ticks=False,
        use_math_text=False,
        reverse=False,
        labelpad=0.0,
        hist_kwargs=None,
        group="posterior",
        var_names=None,
        filter_vars=None,
        coords=None,
        divergences=False,
        divergences_kwargs=None,
        labeller=None,
    )


The Python library ``GetDist <https://getdist.readthedocs.io/en/latest/>``_ can also be used to create plots of the
results. 

This is described in the ``plot`` package of the workspace.

Start Point
-----------

For maximum likelihood estimator (MLE) and Markov Chain Monte Carlo (MCMC) non-linear searches, parameter space
sampling is built around having a "location" in parameter space.

This could simply be the parameters of the current maximum likelihood model in an MLE fit, or the locations of many
walkers in parameter space (e.g. MCMC).

For many model-fitting problems, we may have an expectation of where correct solutions lie in parameter space and
therefore want our non-linear search to start near that location of parameter space. Alternatively, we may want to
sample a specific region of parameter space, to determine what solutions look like there.

The start-point API allows us to do this, by manually specifying the start-point of an MLE fit or the start-point of
the walkers in an MCMC fit. Because nested sampling draws from priors, it cannot use the start-point API.

We now define the start point of certain parameters in the model as follows.

.. code-block:: python

    initializer = af.InitializerParamBounds(
        {
            model.centre: (49.0, 51.0),
            model.normalization: (4.0, 6.0),
            model.sigma: (1.0, 2.0),
        }
    )


Similar behaviour can be achieved by customizing the priors of a model-fit. We could place ``GaussianPrior``'s
centred on the regions of parameter space we want to sample, or we could place tight ``UniformPrior``'s on regions
of parameter space we believe the correct answer lies.

The downside of using priors is that our priors have a direct influence on the parameters we infer and the size
of the inferred parameter errors. By using priors to control the location of our model-fit, we therefore risk
inferring a non-representative model.

For users more familiar with statistical inference, adjusting ones priors in the way described above leads to
changes in the posterior, which therefore impacts the model inferred.

Emcee (MCMC)
------------

The Emcee sampler is a Markov Chain Monte Carlo (MCMC) Ensemble sampler. It is a Python implementation of the
``Goodman & Weare <https://msp.org/camcos/2010/5-1/p04.xhtml>``_ affine-invariant ensemble MCMC sampler.

Information about Emcee can be found at the following links:

- https://github.com/dfm/emcee
- https://emcee.readthedocs.io/en/stable/

The following workspace example shows examples of fitting data with Emcee and plotting the results.

- ``autofit_workspace/notebooks/searches/mcmc/Emcee.ipynb``
- ``autofit_workspace/notebooks/plot/EmceePlotter.ipynb``

The following code shows how to use Emcee with all available options.

.. code-block:: python

    search = af.Emcee(
        nwalkers=30,
        nsteps=1000,
        initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=True,
            check_size=100,
            required_length=50,
            change_threshold=0.01,
        ),
    )

Zeus (MCMC)
-----------

The Zeus sampler is a Markov Chain Monte Carlo (MCMC) Ensemble sampler. 

Information about Zeus can be found at the following links:

- https://github.com/minaskar/zeus
- https://zeus-mcmc.readthedocs.io/en/latest/

.. code-block:: python

    search = af.Zeus(
        nwalkers=30,
        nsteps=1001,
        initializer=af.InitializerBall(lower_limit=0.49, upper_limit=0.51),
        auto_correlation_settings=af.AutoCorrelationsSettings(
            check_for_convergence=True,
            check_size=100,
            required_length=50,
            change_threshold=0.01,
        ),
        tune=False,
        tolerance=0.05,
        patience=5,
        maxsteps=10000,
        mu=1.0,
        maxiter=10000,
        vectorize=False,
        check_walkers=True,
        shuffle_ensemble=True,
        light_mode=False,
    )

DynestyDynamic (Nested Sampling)
--------------------------------

The DynestyDynamic sampler is a Dynamic Nested Sampling algorithm. It is a Python implementation of the
``Speagle <https://arxiv.org/abs/1904.02180>``_ algorithm.

Information about Dynesty can be found at the following links:

- https://github.com/joshspeagle/dynesty
- https://dynesty.readthedocs.io/en/latest/

.. code-block:: python

    search = af.DynestyDynamic(
        nlive=50,
        bound="multi",
        sample="auto",
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        walks=25,
        facc=0.5,
        slices=5,
        fmove=0.9,
        max_move=100,
    )

DynestyStatic (Nested Sampling)
-------------------------------

The DynestyStatic sampler is a Static Nested Sampling algorithm. It is a Python implementation of the
``Speagle <https://arxiv.org/abs/1904.02180>``_ algorithm.

Information about Dynesty can be found at the following links:

- https://github.com/joshspeagle/dynesty
- https://dynesty.readthedocs.io/en/latest/

.. code-block:: python

    search = af.DynestyStatic(
        nlive=50,
        bound="multi",
        sample="auto",
        bootstrap=None,
        enlarge=None,
        update_interval=None,
        walks=25,
        facc=0.5,
        slices=5,
        fmove=0.9,
        max_move=100,
    )

LBFGS
-----

The LBFGS sampler is a Local Optimization algorithm. It is a Python implementation of the scipy.optimize.lbfgs
algorithm.

Information about the L-BFGS method can be found at the following links:

- https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html

.. code-block:: python

    search = af.LBFGS(
        tol=None,
        disp=None,
        maxcor=10,
        ftol=2.220446049250313e-09,
        gtol=1e-05,
        eps=1e-08,
        maxfun=15000,
        maxiter=15000,
        iprint=-1,
        maxls=20,
    )
