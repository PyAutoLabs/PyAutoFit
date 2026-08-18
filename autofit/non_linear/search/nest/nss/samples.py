from autofit.non_linear.samples.nest import SamplesNest


class NSSamples(SamplesNest):
    """Posterior samples from an ``af.NSS`` (Nested Slice Sampling) fit.

    Thin subclass of ``SamplesNest`` that exists primarily for type identity —
    aggregator code and downstream consumers can distinguish an NSS run from
    a Nautilus / Dynesty run by ``isinstance(samples, NSSamples)``. The
    actual posterior + log-evidence wiring is inherited from ``SamplesNest``
    (and ultimately ``SamplesPDF``); ``af.NSS`` builds the ``sample_list`` in
    ``NSS.samples_via_internal_from``.
    """

    @property
    def log_evidence_error(self) -> float:
        """The stochastic batch-error of the NS evidence estimate.

        NSS returns an array of log-evidence estimates ``logZs`` across the
        live ensemble (the Monte Carlo simulation of ``log_dX``). This
        property exposes the standard deviation as the natural per-sample
        uncertainty. The mean is reported as ``log_evidence`` in
        ``samples_info``.
        """
        return float(self.samples_info.get("log_evidence_error", float("nan")))
