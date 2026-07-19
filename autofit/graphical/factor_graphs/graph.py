from collections import Counter, defaultdict
from functools import reduce
from itertools import count
from typing import Tuple, Dict, Collection, List, Type

import numpy as np

from autonerves import cached_property
from autofit.graphical.factor_graphs.abstract import FactorValue, AbstractNode
from autofit.graphical.factor_graphs.factor import Factor
from autofit.graphical.factor_graphs.jacobians import VectorJacobianProduct
from autofit.graphical.utils import rescale_to_artists
from autofit.mapper.variable import Variable


class FactorGraph(AbstractNode):
    def __init__(
            self,
            factors: Collection[Factor],
    ):
        """
        A graph relating factors

        Parameters
        ----------
        factors
            Nodes wrapping individual factors in a model
        """
        self._name = "(%s)" % "*".join(f.name for f in factors)

        self._factors = tuple(factors)

        self._factor_all_variables = {f: f.all_variables for f in self._factors}

        self._call_sequence = self._get_call_sequence()

        self._validate()

        _kwargs = {variable.name: variable for variable in self.variables}

        super().__init__(**_kwargs)

    def copy(self):
        return type(self)(self.factors)

    def related_factors(self, variable: Variable, excluded_factor=None) -> List[Factor]:
        """
        A list of factors which contain the variable.

        Parameters
        ----------
        excluded_factor
            A factor that should be excluded from the list
        variable
            A variable in the graph which will be related to one
            or more factors

        Returns
        -------
        The factors associated with the variable
        """
        return sorted(
            {
                factor
                for factor in self._factors
                if variable in factor.variables
                   and (excluded_factor is None or factor != excluded_factor)
            }
        )

    def _factors_with_type(self, factor_type: Type[Factor]) -> List[Factor]:
        """
        Find all factors with a given type
        """
        return [factor for factor in self._factors if isinstance(factor, factor_type)]

    def factors_by_type(self) -> Dict[Type[Factor], List[Factor]]:
        """
        A dictionary mapping types of factor to all factors of that type
        """
        factors_by_type = defaultdict(list)
        for factor in self._factors:
            factors_by_type[type(factor)].append(factor)
        return factors_by_type

    @property
    def info(self) -> str:
        """
        Describes the graph. Output in graph.info
        """
        string = ""
        for factor_type, factors in self.factors_by_type().items():
            factor_info = "\n\n".join(factor.info for factor in factors)
            string = f"{string}{factor_type.__name__}s\n\n{factor_info}\n\n"
        return string

    def make_results_text(self, model_approx) -> str:
        """
        Generate text describing the graph w.r.t. a given model approximation
        """
        results_text = "\n\n".join(
            factor.make_results_text(model_approx) for factor in self._factors
        )
        return f"{self.name}\n\n{results_text}"

    @property
    def name(self):
        return self._name

    def _validate(self):
        """
        Raises
        ------
        If there is an inconsistency with this graph
        """
        det_var_counts = ", ".join(
            map(
                str,
                [
                    v
                    for v, c in Counter(
                    v for f in self.factors for v in f.deterministic_variables
                ).items()
                    if c > 1
                ],
            )
        )
        if det_var_counts:
            raise ValueError(
                "Improper FactorGraph, "
                f"Deterministic variables {det_var_counts} appear in "
                "multiple factors"
            )

    @cached_property
    def all_variables(self):
        return reduce(
            frozenset.union, (factor.all_variables for factor in self.factors)
        )

    @cached_property
    def deterministic_variables(self):
        return reduce(
            frozenset.union, (factor.deterministic_variables for factor in self.factors)
        )

    @cached_property
    def variables(self):
        return self.all_variables - self.deterministic_variables

    def _get_call_sequence(self) -> List[List[Factor]]:
        """
        Compute the order in which the factors must be evaluated. This is done by checking whether
        all variables required to call a factor are present in the set of variables encapsulated
        by all factors, not including deterministic variables.

        Deterministic variables must be computed before the dependent factors can be computed.
        """
        call_sets = defaultdict(list)
        for factor in self.factors:
            missing_vars = frozenset(factor.variables.difference(self.variables))
            call_sets[missing_vars].append(factor)

        call_sequence = []
        while call_sets:
            # the factors that can be evaluated have no missing variables
            factors = call_sets.pop(frozenset(()))
            # if there's a KeyError then the FactorGraph is improper
            calls = []
            new_variables = set()
            for factor in factors:
                det_vars = factor.deterministic_variables
                calls.append(factor)
                # TODO: this might cause problems
                # TODO: if det_vars appear more than once
                new_variables.update(det_vars)

            call_sequence.append(calls)

            # update to include newly calculated factors
            for missing in list(call_sets.keys()):
                if missing.intersection(new_variables):
                    factors = call_sets.pop(missing)
                    call_sets[missing.difference(new_variables)].extend(factors)

        return call_sequence

    def _call_args(self, *args):
        return self(dict(zip(self.args, args)))

    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
    ) -> FactorValue:
        """
        Call each function in the graph in the correct order, adding the logarithmic results.

        Deterministic values computed in initial factor calls are added to a dictionary and
        passed to subsequent factor calls.

        Parameters
        ----------
        variable_dict
            Positional arguments
        axis
            Keyword arguments

        Returns
        -------
        Object comprising the log value of the computation and a dictionary containing
        the values of deterministic variables.
        """

        # generate set of factors to call, these are indexed by the
        # missing deterministic variables that need to be calculated
        log_value = 0.0
        det_values = {}
        variables = {**self.fixed_values, **variable_dict}

        missing = set(v.name for v in self.variables).difference(
            v.name for v in variables
        )
        if missing:
            n_miss = len(missing)
            missing_str = ", ".join(missing)
            raise ValueError(
                f"{self} missing {n_miss} arguments: {missing_str}"
                f"factor graph call signature: {self.call_signature}"
            )

        for calls in self._call_sequence:
            # TODO parallelise this part?
            for factor in calls:
                ret = factor(variables)
                log_value += np.sum(ret)
                det_values.update(ret.deterministic_values)
                variables.update(ret.deterministic_values)

        return FactorValue(log_value, det_values)

    def func_jacobian(
            self, variable_dict: Dict[Variable, np.ndarray], **kwargs
    ) -> FactorValue:

        # generate set of factors to call, these are indexed by the
        # missing deterministic variables that need to be calculated
        log_value = 0.0
        det_values = {}
        factor_jacs = {}
        variables = {**self.fixed_values, **variable_dict}

        missing = set(v.name for v in self.variables).difference(
            v.name for v in variables
        )
        if missing:
            n_miss = len(missing)
            missing_str = ", ".join(missing)
            raise ValueError(
                f"{self} missing {n_miss} arguments: {missing_str}"
                f"factor graph call signature: {self.call_signature}"
            )

        for calls in self._call_sequence:
            # TODO parallelise this part?
            for factor in calls:
                ret, factor_jacs[factor] = factor.func_jacobian(variables, **kwargs)
                log_value += ret

                det_values.update(ret.deterministic_values)
                variables.update(ret.deterministic_values)

        jac_out = (FactorValue,) + tuple(det_values)
        jac_variables = tuple(variables)

        def graph_vjp(args):
            args = args if isinstance(args, tuple) else (args,)
            grads = {}
            grads.update(zip(jac_out, args))
            for calls in self._call_sequence[::-1]:
                for factor in calls:
                    factor_grad = factor_jacs[factor](grads)
                    for v, v_grad in factor_grad.items():
                        grads[v] = grads.get(v, 0) + v_grad

            return tuple(grads[v] for v in jac_variables)

        fval = FactorValue(log_value, det_values)
        graph_vjp = VectorJacobianProduct(
            jac_out,
            graph_vjp,
            *jac_variables,
            out_shapes=fval.to_dict().shapes,
        )

        return fval, graph_vjp

    def __mul__(self, other: AbstractNode) -> "FactorGraph":
        """
        Combine this object with another factor node or graph, creating
        a new graph that comprises all of the factors of the two objects.
        """
        factors = self.factors

        if isinstance(other, FactorGraph):
            factors += other.factors
        elif isinstance(other, Factor):
            factors += (other,)
        else:
            raise TypeError(
                f"type of passed element {(type(other))} "
                "does not match required types, (`FactorGraph`, `FactorNode`)"
            )

        return type(self)(factors)

    def __repr__(self) -> str:
        factors_str = " * ".join(map(repr, self.factors))
        return f"({factors_str})"

    @property
    def factors(self) -> Tuple[Factor, ...]:
        return self._factors

    @property
    def factor_all_variables(self) -> Dict[Factor, List[Variable]]:
        return self._factor_all_variables

    @property
    def graph(self):
        try:
            import networkx as nx
        except ImportError as e:
            raise ImportError("networkx required for graph") from e

        G = nx.Graph()
        G.add_nodes_from(self.factors, bipartite="factor")
        G.add_nodes_from(self.all_variables, bipartite="variable")
        G.add_edges_from(
            (f, v) for f, vs in self.factor_all_variables.items() for v in vs
        )

        return G

    def draw_graph(
            self,
            pos=None,
            ax=None,
            size=20,
            color="k",
            fill="w",
            factor_shape="s",
            variable_shape="o",
            factor_labels=None,
            variable_labels=None,
            factor_kws=None,
            variable_kws=None,
            edge_kws=None,
            factors=None,
            draw_labels=False,
            label_kws=None,
            **kwargs,
    ):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "Matplotlib and networkx required for draw_graph()"
            ) from e
        except RuntimeError as e:
            print("Matplotlib unable to open display")
            raise e

        if ax is None:
            ax = plt.gca()

        G = self.graph
        if pos is None:
            pos = bipartite_layout(factors or self.factors)

        kwargs.setdefault("ms", size)
        kwargs.setdefault("c", color)
        kwargs.setdefault("mec", color)
        kwargs.setdefault("mfc", fill)
        kwargs.setdefault("ls", "")

        factor_kws = factor_kws or {}
        factor_kws.setdefault("marker", factor_shape)

        variable_kws = variable_kws or {}
        variable_kws.setdefault("marker", variable_shape)

        # draw factors
        xy = np.array([pos[f] for f in self.factors]).T
        fs = ax.plot(*xy, **{**kwargs, **factor_kws})
        # draw variables
        xy = np.array([pos[f] for f in self.all_variables]).T
        vs = ax.plot(*xy, **{**kwargs, **variable_kws})
        # draw edges
        edges = nx.draw_networkx_edges(G, pos, **(edge_kws or {}))

        # remove ticks from axes
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        if draw_labels:
            self.draw_graph_labels(
                pos,
                ax=ax,
                factor_labels=factor_labels,
                variable_labels=variable_labels,
                **(label_kws or {})
            )
        return pos, fs, vs, edges

    def draw_graph_labels(
            self,
            pos,
            factor_labels=None,
            variable_labels=None,
            shift=0.1,
            f_shift=None,
            v_shift=None,
            f_horizontalalignment="right",
            v_horizontalalignment="left",
            f_kws=None,
            v_kws=None,
            graph=None,
            ax=None,
            rescale=True,
            **kwargs,
    ):
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError as e:
            raise ImportError(
                "Matplotlib and networkx required for draw_graph()"
            ) from e
        ax = ax or plt.gca()
        graph = graph or self.graph
        factor_labels = factor_labels or {f: f.name for f in self.factors}
        variable_labels = variable_labels or {v: v.name for v in self.all_variables}
        f_kws = f_kws or {"horizontalalignment": f_horizontalalignment}
        v_kws = v_kws or {"horizontalalignment": v_horizontalalignment}

        f_shift = f_shift or shift
        f_pos = {f: (x - f_shift, y) for f, (x, y) in pos.items()}
        v_shift = v_shift or shift
        v_pos = {f: (x + v_shift, y) for f, (x, y) in pos.items()}

        text = {
            **nx.draw_networkx_labels(
                graph, f_pos, labels=factor_labels, ax=ax, **f_kws, **kwargs
            ),
            **nx.draw_networkx_labels(
                graph, v_pos, labels=variable_labels, ax=ax, **v_kws, **kwargs
            ),
        }
        if rescale:
            rescale_to_artists(text.values(), ax=ax)

        return text


def bipartite_layout(factors):
    n_factors = len(factors)
    n_variables = len(set().union(*(f.variables for f in factors)))
    n = max(n_factors, n_variables)
    factor_count = count()
    variable_count = count()

    pos = {}
    for factor in factors:
        pos[factor] = 0, next(factor_count) * n / n_factors
        for v in factor.variables:
            if v not in pos:
                pos[v] = 1, next(variable_count) * n / n_variables

    return pos
