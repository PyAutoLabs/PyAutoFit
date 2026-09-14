# PyAutoFit: Scientific Inference with Natural Language

[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
[![Python Versions](https://img.shields.io/pypi/pyversions/autofit)](https://pypi.org/project/autofit/)
[![PyPI Version](https://img.shields.io/pypi/v/autofit.svg)](https://pypi.org/project/autofit/)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.14.1/notebooks/overview/overview_1_the_basics.ipynb)
[![Tests](https://github.com/PyAutoLabs/PyAutoFit/actions/workflows/main.yml/badge.svg)](https://github.com/PyAutoLabs/PyAutoFit/actions)
[![Build](https://github.com/PyAutoLabs/PyAutoHands/actions/workflows/release.yml/badge.svg)](https://github.com/PyAutoLabs/PyAutoHands/actions)
[![Documentation Status](https://readthedocs.org/projects/pyautofit/badge/?version=latest)](https://pyautofit.readthedocs.io/en/latest/?badge=latest)
[![JOSS](https://joss.theoj.org/papers/10.21105/joss.02550/status.svg)](https://doi.org/10.21105/joss.02550)

[AI Assistant](https://github.com/PyAutoLabs/autofit_assistant) |
[Documentation](https://pyautofit.readthedocs.io/en/latest/index.html) |
[Installation Guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) |
[Introduction on Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.14.1/notebooks/overview/overview_1_the_basics.ipynb) |
[HowToFit](https://github.com/PyAutoLabs/HowToFit)

**PyAutoFit is designed so scientists can bring their models, data and likelihood code, then fit models, explore results and develop analyses using natural language with an AI coding agent.**

## Getting Started

**PyAutoFit** and the **autofit_assistant** allow one to perform scientific inference using purely natural language.
Simply open your AI coding agent (`codex` or `Claude Code` are recommended) and input the following prompt:

<sub><b>Example Natural Language Prompt for Claude Code, Codex or other AI coding agent</b></sub>

```text
I want to perform scientific inference with PyAutoFit (https://github.com/PyAutoLabs/PyAutoFit) and the
autofit_assistant (https://github.com/PyAutoLabs/autofit_assistant).

Begin the "start here" guide for a new user.
```

**PyAutoFit** is designed to be used entirely through natural language — the
[natural-language inference page](https://pyautofit.readthedocs.io/en/latest/overview/natural_language.html)
walks through this, including its **AI First Design**.

## Bring Your Own Likelihood (BYOL)

Already have a likelihood function for your science problem? **Point the assistant at your existing code and it can
set it up with PyAutoFit** — defining the model, choosing priors with you, configuring a search and organising the
results:

<sub><b>Example Natural Language Prompt for Claude Code, Codex or other AI coding agent</b></sub>

```text
Set up PyAutoFit with my existing science project. An example likelihood function can be found at
[GitHub link or local directory].

First, give me an overview of my project and likelihood function. Compose an appropriate model, explain it to me,
and recommend a non-linear search (for example MCMC, nested sampling or maximum-likelihood estimation).

Do not begin inference until we have discussed the setup and I give you the go-ahead.

Once inference is running, explain how the results are written to disk and show me how to inspect and interpret
them with PyAutoFit.
```

The assistant answers this prompt with its
[**BYOL** mode](https://github.com/PyAutoLabs/autofit_assistant/blob/main/modes/byol.md) — bring your own likelihood,
the assistant brings the inference. It reads your code and restates what your likelihood scores, composes a model with
priors chosen with you, wraps and validates your function unchanged, recommends a search, and runs nothing until you
say go.

Your existing science code remains the source of the likelihood. With PyAutoFit built around it, you can perform
inference through natural language while gaining access to features such as flexible priors and model composition,
MCMC and nested sampling, automated result handling, model comparison and scalable workflows.

## What is PyAutoFit?

**PyAutoFit** is a domain-agnostic Python package for scientific model fitting and Bayesian inference. It supports
nested sampling, MCMC and optimisation, alongside advanced methods such as hierarchical models, search chaining
and Bayesian model comparison.

[**autofit_assistant**](https://github.com/PyAutoLabs/autofit_assistant) connects natural-language requests to runnable
**PyAutoFit** workflows. Ask it to compose a model, discuss priors, run inference or compare competing explanations.
You guide the science; it writes and runs Python scripts you can inspect, rerun and share.

## Human Readable Documentation

- [The PyAutoFit readthedocs](https://pyautofit.readthedocs.io/en/latest), which includes an [installation guide](https://pyautofit.readthedocs.io/en/latest/installation/overview.html) and an overview of **PyAutoFit**'s core features.
- [The introduction Jupyter Notebook on Colab](https://colab.research.google.com/github/PyAutoLabs/autofit_workspace/blob/2026.9.14.1/notebooks/overview/overview_1_the_basics.ipynb), where you can try **PyAutoFit** in a web browser (without installation).
- [The autofit_workspace GitHub repository](https://github.com/PyAutoLabs/autofit_workspace), which includes example scripts demonstrating **PyAutoFit**'s features.
- [The standalone HowToFit repository](https://github.com/PyAutoLabs/HowToFit), a series of Jupyter notebook lectures which give new users a step-by-step introduction to **PyAutoFit**.

## HowToFit

For users less familiar with Bayesian inference and scientific analysis you may wish to read through
the **HowToFit** lectures. These teach you the basic principles of Bayesian inference, with the
content pitched at undergraduate level and above.

The lectures are available in the [standalone HowToFit repository](https://github.com/PyAutoLabs/HowToFit).

Use the [Jupyter notebooks](https://github.com/PyAutoLabs/HowToFit/tree/main/notebooks) if you want to run
the code (recommended), or read the available [Markdown lectures](https://github.com/PyAutoLabs/HowToFit/tree/main/markdown)
directly on GitHub.

For help alongside the lectures, open the [autofit_assistant](https://github.com/PyAutoLabs/autofit_assistant)
repository in your AI coding agent, following its setup instructions, and paste:

```text
Enter HowToFit mode.

I want to work through the HowToFit lectures. Show me where to find them
and how to use Jupyter Notebook or Markdown, then help me with questions
as I go.
```

The assistant will answer questions about concepts, equations, code and results as you study, and help with
notebook errors. Share the lecture link and section or the cell you are working on; you choose when to move on.

## Support

Support for installation issues, help with Fit modeling and using **PyAutoFit** is available by
[raising an issue on the GitHub issues page](https://github.com/PyAutoLabs/PyAutoFit/issues).

We also offer support on the **PyAutoFit** [Slack channel](https://pyautoFit.slack.com/), where we also provide the
latest updates on **PyAutoFit**. Slack is invitation-only, so if you'd like to join send
an [email](https://github.com/Jammy2211) requesting an invite.
