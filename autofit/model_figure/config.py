"""
The single gate every model figure is written behind.

``output.yaml``'s ``model_figure`` key is read **strictly** here so that both
the per-search ``model.png`` hook (``autofit.non_linear.paths.directory``) and
the EP factor-graph figures ask the same question of the same key.
"""

from autonerves import conf

__all__ = ["model_figure_enabled"]


def model_figure_enabled() -> bool:
    """
    Whether `model.png` -- the model figure drawn beside `model.info` -- is written.

    The `output.yaml` key `model_figure` is read **strictly**: an absent key means
    off.

    This deliberately does NOT use `autonerves.output.should_output`
    (`autonerves/output.py:should_output`), which falls back to `output.yaml`'s
    `default:` entry whenever a key is absent. That entry is `true` in every
    workspace, so a brand new key that so far only exists in the library's own
    default config would silently switch the figure ON in every config which has
    not yet added it. Only a config that explicitly opts in writes the file.
    """
    try:
        return bool(conf.instance["output"]["model_figure"])
    except KeyError:
        return False
