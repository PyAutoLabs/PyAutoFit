from autofit.mapper.prior.arithmetic import ArithmeticMixin
from autofit.mapper.prior_model.prior_model import Model, Prior


class AnnotationPriorModel(Model, ArithmeticMixin):
    def __init__(self, cls, parent_class, true_argument_name, **kwargs):
        self.parent_class = parent_class
        self.true_argument_name = true_argument_name
        self._value = None
        super().__init__(cls, **kwargs)

    def make_prior(self, attribute_name):
        if self._value is None:
            self._value = Prior.for_class_and_attribute_name(
                self.parent_class, self.true_argument_name
            )
        return self._value
