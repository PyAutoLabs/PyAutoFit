.. _howtofit:

HowToFit Lectures
=================

To learn how to use **PyAutoFit**, the best starting point is the **HowToFit** lecture series, which are found on
the `autofit_workspace <https://github.com/Jammy2211/autofit_workspace>`_ and at
our `colab <https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.4.13.6/start_here.ipynb>`_.

The lectures are provided as *Jupyter notebooks* and currently consist of 3 chapters:

- ``chapter_1_introduction``: Introduction lectures describing how to compose and fit a models.
- ``chapter_2_scientific_workflow``: Not written yet, but will describe how to build a scientific workflow as described the overview example.
- ``chapter_3_graphical_models``: How to compose and fit graphical models which fit many datasets simultaneously.

Statistics and Theory
---------------------

**HowToFit** assumes minimal previous knowledge of statistics, model-fitting and Bayesian inference. However, it is beneficial
yourself a basic theoretical grounding as you go through the lectures. I heartily recommend you aim to read up on
the concepts introduced throughout the lectures once you understand how to use them in **PyAutoFit**.

Jupyter Notebooks
-----------------

The tutorials are supplied as *Jupyter notebooks*, which come with a ``.ipynb`` suffix. For those new to
Python, *Jupyter notebooks* are a different way to write, view and use Python code. Compared to the
traditional Python scripts, they allow:

- Small blocks of code to be viewed and run at a time
- Images and visualization from a code to be displayed directly underneath it.
- Text script to appear between the blocks of code.

This makes them an ideal way for us to present the HowToFit lecture series, therefore I recommend you get
yourself a Jupyter notebook viewer (https://jupyter.org/) if you havent done so already.

If you *really* want to use Python scripts, all tutorials are supplied a ``.py`` python files in the ``scripts``
folder of each chapter.

For actual **PyAutoFit** use I recommend you use Python scripts. Therefore, as you go through the lecture
series you will notice that we will transition you to Python scripts.

Code Style and Formatting
-------------------------

You may notice the style and formatting of our Python code looks different to what you are used to. For
example, it is common for brackets to be placed on their own line at the end of function calls, the inputs
of a function or class may be listed over many separate lines and the code in general takes up a lot more
space then you are used to.

This is intentional, because we believe it makes the cleanest, most readable code possible. In fact, lots
of people do, which is why we use an auto-formatter to produce the code in a standardized format. If you're
interested in the style and would like to adapt it to your own code, check out the Python auto-code formatter
``black``.

https://github.com/python/black

How to Approach HowToFit
------------------------

The **HowToFit** lecture series current sits at 3 chapters, and each will take more than a couple of hours to go through
properly. You probably want to be begin modeling with **PyAutoFit** faster than that! Furthermore, the concepts in the
later chapters are pretty challenging, and familiarity with **PyAutoFit** and model fitting is desirable before you
tackle them.

Therefore, we recommend that you complete chapter 1 and then apply what you've learnt to a model fitting problem you are
interested in, building on the scripts found in the 'autofit_workspace/examples' folder. Once you're happy
with the results and confident with your use of **PyAutoFit**, you can then begin to cover the advanced functionality
covered in other chapters.

Overview of Chapter 1: Introduction
-----------------------------------

In chapter 1, we'll learn about model composition and fitting and **PyAutoFit**:

- ``tutorial_1_models.py``: What a probabilistic model is and how to compose a model using PyAutoFit.
- ``tutorial_2_fitting_data.py``: Fitting a model to data and quantifying its goodness-of-fit.
- ``tutorial_3_non_linear_search.py``: Searching a non-linear parameter spaces to find the model that best fits the data.
- ``tutorial_4_why_modeling_is_hard.py``: Composing more complex models in a scalable and extensible way.
- ``tutorial_5_results_and_samples.py``: Interpreting the results of a model-fit and using the samples to perform scientific analysis.
- ``tutorial_6_multiple_datasets.py``: Fitting multiple datasets simultaneously and how to compose models that are shared between datasets.
- ``tutorial_7_bayesian_inference``: A formal introduction to Bayesian inference and how it is used to perform model-fitting.
- ``tutorial_8_astronomy_example.py``: An example of how PyAutoFit can be used to model-fit astronomical data.

Overview of Chapter 2: Scientific Workflow
-------------------------------------------

Chapter 2 is not written yet, but will cover the following topics:

- ``tutorial_1_output``: Outputting as much information as possible about the model-fit to the hard-disk.
- ``tutorial_2_loading_results``: Loading the results of a model-fit from the hard-disk (with and without database).
- ``tutorial_3_model_customization``: Customizing the model in a way that enables more detailed analysis of the results.
- ``tutorial_4_searches``: Customizing the search used to perform the model-fit.
- ``tutorial_5_config``: Customizing the configuration of the model-fit.
- ``tutorial_6_latent_variables``: Introducing latent variables and how they can be used to model complex datasets.
- ``tutorial_7_astronomy_example``: An example of how PyAutoFit can be used to model-fit astronomical data.

Overview of Chapter 3: Graphical Models
---------------------------------------

- ``tutorial_1_individual_models``: Inferring global parameters from a dataset by fitting the model to each individual dataset one-by-one.
- ``tutorial_2_graphical_model``: Fitting the dataset with a graphical model that fits all datasets simultaneously to infer the global parameters.
- ``tutorial_3_graphical_benefits``: Illustrating the benefits of graphical modeling over fitting individual datasets one-by-one.
- ``tutorial_4_hierarchical_models``: Fitting hierarchical models using the graphical modeling framework.
- ``tutorial_5_expectation_propagation``: Scaling graphical models up to fit extremely large datasets using Expectation Propagation (EP).