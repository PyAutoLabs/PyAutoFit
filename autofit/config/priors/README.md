The prior config files contain the default priors and related variables for every model component when it is used as a
model.

They appear as follows:

```bash
Gaussian:
  centre:
    type: Uniform
    lower_limit: 0.0
    upper_limit: 100.0
    width_modifier:
      type: Absolute
      value: 20.0
    limits:
      lower: -inf
      upper: inf
```

The sections of this example config set the following:

> type {Uniform, Gaussian, LogUniform}
>
> : The default prior given to this parameter when used by the non-linear search. In the example above, a
>   UniformPrior is used with lower_limit of 0.0 and upper_limit of 4.0. A GaussianPrior could be used by
>   putting "Gaussian" in the "type" box, with "mean" and "sigma" used to set the default values. Any prior can be
>   set in an analogous fashion (see the example configs).
>
> width_modifier
>
> : When the results of a search are passed to a subsequent search to set up the priors of its non-linear search,
>   this entry describes how the Prior is passed. For a full description of prior passing, checkout the examples
>   in 'autolens_workspace/examples/complex/linking'.
>
> limits
>
> : When the results of a search are passed to a subsequent search, they are passed using a GaussianPrior. The
>   limits set the physical lower and upper limits of this GaussianPrior, such that parameter samples
>   can not go beyond these limits.

The files `template_module.yaml` and `TemplateObject.yaml` give templates one can use to set up prior default
configs for your own model components.
