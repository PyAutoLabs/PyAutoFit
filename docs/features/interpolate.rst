.. _interpolate:

Model Interpolation
===================

It is common to fit a model to many similar datasets, where it is anticipated that one or more model parameters vary
smoothly across the datasets.

For example, the datasets may be taken at different times, where the signal in the data and therefore model parameters
vary smoothly as a function of time. Alternatively, the datasets may be taken at different wavelengths, with the signal
varying smoothly as a function of wavelength.

In any of these cases, it may be desireable to fit the datasets one-by-one and then interpolate the results in order
to determine the most likely model parameters at any point in time (or at any wavelength).

This example illustrates model interpolation functionality in **PyAutoFit** using the example of fitting 3 noisy
1D Gaussians, where these data are assumed to have been taken at 3 different times. The ``centre`` of each ``Gaussian``
varies smoothly over time. The interpolation is therefore used to estimate the ``centre`` of each ``Gaussian`` at any time
outside of the times the data were observed.

Data
----

We illustrate model interpolation using 3 noisy 1D Gaussian datasets taken at 3 different times, where the ``centre`` of
each ``Gaussian`` varies smoothly over time.

The datasets are taken at 3 times, t=0, t=1 and t=2.

.. code-block:: python

    total_datasets = 3

    data_list = []
    noise_map_list = []
    time_list = []

    for time in range(3):
        dataset_name = f"time_{time}"

        dataset_prefix_path = Path("dataset/example_1d/gaussian_x1_variable")
        dataset_path = dataset_prefix_path / dataset_name

        data = af.util.numpy_array_from_json(file_path=dataset_path / "data.json")
        noise_map = af.util.numpy_array_from_json(file_path=dataset_path / "noise_map.json")

        data_list.append(data)
        noise_map_list.append(noise_map)
        time_list.append(time)

Visual comparison of the datasets shows that the ``centre`` of each ``Gaussian`` varies smoothly over time, with it moving
from pixel 40 at t=0 to pixel 60 at t=2.

.. image:: https://raw.githubusercontent.com/rhayes777/PyAutoFit/main/docs/features/images/hi.png
  :width: 600
  :alt: Alternative text

Fit
---

We now fit each of the 3 datasets.

The fits are performed in a for loop, with the docstrings inside the loop explaining the code.

The interpolate at the end of the fits uses the maximum log likelihood model of each fit, which we store in a list.

.. code-block:: python

    ml_instances_list = []

    for data, noise_map, time in zip(data_list, noise_map_list, time_list):

        """
        __Analysis__

        For each dataset we create an ``Analysis`` class, which includes the ``log_likelihood_function`` we fit the data with.
        """
        analysis = af.ex.Analysis(data=data, noise_map=noise_map)

        """
        __Time__

        The model composed below has an input not seen in other examples, the parameter ``time``.

        This is the time that the simulated data was acquired and is not a free parameter in the fit.

        For interpolation it plays a crucial role, as the model is interpolated to the time of every dataset input
        into the model below. If the ``time`` input were missing, interpolation could not be performed.

        Over the iterations of the for loop, the ``time`` input will therefore be the values 0.0, 1.0 and 2.0.

        __Model__

        We now compose our model, which is a single ``Gaussian``.

        The ``centre`` of the ``Gaussian`` is a free parameter with a ``UniformPrior`` that ranges between 0.0 and 100.0.

        We expect the inferred ``centre`` inferred from the fit to each dataset to vary smoothly as a function of time.
        """
        model = af.Collection(
            gaussian=af.Model(af.ex.Gaussian),
            time=time
        )

        """
        __Search__

        The model is fitted to the data using the nested sampling algorithm
        Dynesty (https://dynesty.readthedocs.io/en/latest/).
        """
        search = af.DynestyStatic(
            path_prefix=path.join("interpolate"),
            name=f"time_{time}",
            nlive=100,
        )

        """
        __Model-Fit__

        We can now begin the model-fit by passing the model and analysis object to the search, which performs a non-linear
        search to find which models fit the data with the highest likelihood.
        """
        result = search.fit(model=model, analysis=analysis)

        """
        __Instances__

        Interpolation uses the maximum log likelihood model of each fit to build an interpolation model of the model as a
        function of time.

        We therefore store the maximum log likelihood model of every fit in a list, which is used below.
        """
        ml_instances_list.append(result.instance)

Interpolation
-------------

Now all fits are complete, we use the ``ml_instances_list`` to build an interpolation model of the model as a function
of time.

This is performed using the ``LinearInterpolator`` object, which interpolates the model parameters as a function of
time linearly between the values computed by the model-fits above.

More advanced interpolation schemes are available and described in the ``interpolation.py`` example.

.. code-block:: python

    interpolator = af.LinearInterpolator(instances=ml_instances_list)

The model can be interpolated to any time, for example time=1.5.

This returns a new ``instance`` of the model, as an instance of the ``Gaussian`` object, where the parameters are computed
by interpolating between the values computed above.

.. code-block:: python

    instance = interpolator[interpolator.time == 1.5]

The ``centre`` of the ``Gaussian`` at time 1.5 is between the value inferred for the first and second fits taken
at times 1.0 and 2.0.

This is a ``centre`` close to a value of 55.0.

.. code-block:: python

    print(f"Gaussian centre of fit 1 (t = 1): {ml_instances_list[0].gaussian.centre}")
    print(f"Gaussian centre of fit 2 (t = 2): {ml_instances_list[1].gaussian.centre}")

    print(f"Gaussian centre interpolated at t = 1.5 {instance.gaussian.centre}")

Serialisation
-------------

The interpolator and model can be serialized to a .json file using **PyAutoConf**'s dedicated serialization methods.

This means an interpolator can easily be loaded into other scripts.

.. code-block:: python

    from autoconf.dictable import output_to_json, from_json

    json_file = path.join(dataset_prefix_path, "interpolator.json")

    output_to_json(obj=interpolator, file_path=json_file)

    interpolator = from_json(file_path=json_file)

Database
--------

It may be inconvenient to fit all the models in a single Python script (e.g. the model-fits take a long time and you
are fitting many datasets).

PyAutoFit's allows you to store the results of model-fits from hard-disk.

Database functionality then allows you to load the results of the fit above, set up the interpolator and perform the
interpolation.

If you are not familiar with the database API, you should checkout the ``cookbook/database.ipynb`` example.

.. code-block:: python

    from autofit.aggregator.aggregator import Aggregator

    agg = Aggregator.from_directory(
        directory=path.join("output", "interpolate"), completed_only=False
    )

    ml_instances_list = [samps.max_log_likelihood() for samps in agg.values("samples")]

    interpolator = af.LinearInterpolator(instances=ml_instances_list)

    instance = interpolator[interpolator.time == 1.5]

    print(f"Gaussian centre of fit 1 (t = 1): {ml_instances_list[0].gaussian.centre}")
    print(f"Gaussian centre of fit 2 (t = 2): {ml_instances_list[1].gaussian.centre}")

    print(f"Gaussian centre interpolated at t = 1.5 {instance.gaussian.centre}")