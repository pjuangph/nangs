import numpy as np
import torch
from .boco import Boco


class Dirichlet(Boco):    
    def __init__(self, sampler, output_fn, name="dirichlet"):
        """Initializes Dirichlet boundary condition. 
            This boundary condition specifies that the value of the outputs have to be equal to a value along the boundaries. 
            For 2D the boundaries are [:,0], [:,-1], [0,:], [-1,:] so if your quantity is f like for laplace then f at boundaries equals to a constant, hence boundary condition of the first order, a constant. 


        More Info: 
            Boundary Conditions: https://www.simscale.com/docs/simwiki/numerics-background/what-are-boundary-conditions/
            Laplace Equation: https://en.wikipedia.org/wiki/Laplace%27s_equation

        Args:
            sampler ([type]): [description]
            output_fn ([type]): [description]
            name (str, optional): [description]. Defaults to "dirichlet".
        """
        super().__init__(name)
        self.vars = sampler.vars
        self.sampler = sampler
        assert callable(output_fn), 'output_fn must be callable'
        self.output_fn = output_fn

    def validate(self, inputs, outputs):
        super().validate()
        assert inputs == self.vars, f'Boco {self.name} with different inputs !'
        _outputs = tuple(self.output_fn(self.sampler.sample(1)).keys())
        if outputs != _outputs:
            print(
                f'Boco {self.name} with different outputs ! {outputs} vs {_outputs}')
        # puedo fitear solo algunos outputs
        # self.output_ids = [outputs.index(v)
        #                    for v in self.vars[1] if v in outputs]

    def sample(self, n_samples=None):
        inputs = self.sampler.sample(n_samples)
        outputs = self.output_fn(inputs)
        return inputs, outputs

    def computeLoss(self, model:torch.nn.Module, criterion, inputs:Tuple[str], outputs:Tuple[str]):
        """This function computes the loss from the model. The model is expected to predict values for all states of the PDE so if the PDE takes x,y,t and predicts phi. Then the model will predict the value of phi at x,y,t for all time.

        Example:
            This function takes the boundary conditions so if you specify the initial condition at u = something for t = 0. 
            It checks how well the model is at predicting u

        Args:
            model (torch.nn.Module): torch model
            criterion (torch.nn._Loss): loss criterion like MSE
            inputs (Tuple[str]): Tuple containing keys for the input variables e.g. ('x','y','t')
            outputs (Tuple[str]): Tuple containing keys for the output variables e.g. ('u','v')

        Returns:
            Dict[str,torch.Tensor]: Dictionary containing the name of the boundary condition and the loss value
        """
        _X, _y = self.sample()
        X = torch.stack([ _X[var] for var in inputs], axis=-1)
        y_hat = model(X)
        __y = []
        for i, var in enumerate(outputs):
            if var in _y:
                __y.append(_y[var])
            else:
                __y.append(y_hat[:, i])
        y = torch.stack(__y, axis=-1)

        return {self.name: criterion(y, y_hat)}
