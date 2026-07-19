from typing import Dict, List

from autonerves import cached_property

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.mapper.prior.uniform import UniformPrior
from autofit.mapper.prior.gaussian import GaussianPrior
from autofit.mapper.prior.log_gaussian import LogGaussianPrior
from autofit.mapper.prior.log_uniform import LogUniformPrior
from autofit.mapper.prior.truncated_gaussian import TruncatedGaussianPrior
from autofit.mapper.prior_model.prior_model import ModelObject
from autofit.mapper.prior_model.prior_model import Model
from autofit.mapper.prior_model.collection import Collection


def str_for_object(obj: ModelObject) -> str:
    """
    Get a string representation of an object, including its id, type
    and important parameters. Used as labels for nodes.

    Parameters
    ----------
    obj
        Some component of the model.

    Returns
    -------
    A string representation of the object.
    """
    if isinstance(obj, Collection):
        return f"{obj.id}:Collection({len(obj)})"
    if isinstance(obj, Model):
        return f"{obj.id}:Model({obj.cls.__name__})"
    if isinstance(obj, UniformPrior):
        return f"{obj.id}:UniformPrior({obj.lower_limit}, {obj.upper_limit})"
    if isinstance(obj, GaussianPrior):
        return f"{obj.id}:GaussianPrior({obj.mean}, {obj.sigma})"
    if isinstance(obj, LogGaussianPrior):
        return f"{obj.id}:LogGaussianPrior({obj.mean}, {obj.sigma})"
    if isinstance(obj, LogUniformPrior):
        return f"{obj.id}:LogUniformPrior({obj.lower_limit}, {obj.upper_limit})"
    if isinstance(obj, TruncatedGaussianPrior):
        return f"{obj.id}:TruncatedGaussianPrior({obj.mean}, {obj.sigma}, ({obj.lower_limit}, {obj.upper_limit})"

    return repr(obj)


def generate_n_colors(n: int) -> List[str]:
    """
    Generate n distinct colors.
    """
    import colorsys

    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hls_to_rgb(hue, 0.5, 1.0)
        hex_color = "#{:02x}{:02x}{:02x}".format(
            int(r * 255), int(g * 255), int(b * 255)
        )
        colors.append(hex_color)
    return colors


class VisualiseGraph:
    def __init__(self, model: AbstractPriorModel):
        """
        Visualise the graph of a model. Models and Priors are nodes
        and edges are the relationships between them.

        Parameters
        ----------
        model
            The model to visualise.
        """
        self.model = model

    def graph(self) -> "nx.DiGraph":
        """
        Generate a graph of the model with just edges. Could
        be superseded by the network method.
        """
        import networkx as nx

        graph = nx.DiGraph()

        def add_model(model):
            model_name = str_for_object(model)

            for name, prior in model.direct_prior_tuples:
                prior_name = str_for_object(prior)
                graph.add_edge(
                    model_name,
                    prior_name,
                    label=name,
                )

            for name, child_model in model.direct_prior_model_tuples:
                add_model(child_model)
                graph.add_edge(
                    model_name,
                    str_for_object(child_model),
                    label=name,
                )

        add_model(self.model)

        return graph

    @cached_property
    def colours(self) -> Dict[type, str]:
        """
        Generate a dictionary of colours for each type of object.
        """
        types = {
            Collection,
            UniformPrior,
            GaussianPrior,
            LogGaussianPrior,
            LogUniformPrior,
            TruncatedGaussianPrior,
        } | {model.cls for _, model in self.model.attribute_tuples_with_type(Model)}
        if isinstance(self.model, Model):
            types.add(self.model.cls)
        return {
            type_: color
            for type_, color in zip(
                sorted(types, key=str),
                generate_n_colors(len(types)),
            )
        }

    def network(self, notebook: bool = False) -> "Network":
        """
        Generate a network of the model including stylised nodes and edges.

        Models and Priors are nodes and edges are the relationships between them.

        Parameters
        ----------
        notebook
            Whether to display the network in a notebook. Must be true when calling
            the show method.

        Returns
        -------
        A network of the model.
        """
        from pyvis.network import Network

        net = Network(
            notebook=notebook,
            directed=True,
            cdn_resources="remote",
        )

        def add_model(obj, **kwargs):
            net.add_node(
                str_for_object(obj),
                shape="square",
                color=self.colours[obj.cls],
                size=15,
                **kwargs,
            )

        def add_collection(obj, **kwargs):
            net.add_node(
                str_for_object(obj),
                shape="hexagon",
                color=self.colours[Collection],
                size=15,
                **kwargs,
            )

        if isinstance(self.model, Model):
            add_model(self.model, borderWidth=5)
        else:
            add_collection(self.model, borderWidth=5)

        for _, model in self.model.attribute_tuples_with_type(Model):
            add_model(model)
        for _, collection in self.model.attribute_tuples_with_type(
            Collection,
            ignore_children=False,
        ):
            add_collection(collection)
        for _, prior in self.model.prior_tuples:
            net.add_node(
                str_for_object(prior),
                shape="dot",
                color=self.colours[type(prior)],
                size=10,
            )

        net.from_nx(self.graph())

        return net

    def save(self, path: str):
        """
        Save the network to a file. This should be an html file.

        Parameters
        ----------
        path
            The path to save the network to.
        """
        self.network().save_graph(path)

    def show(self, name: str):
        """
        Show the network.
        """
        self.network(notebook=True).show(f"{name}.html")
