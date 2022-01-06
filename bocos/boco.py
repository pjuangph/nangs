import torch


class Boco():
    def __init__(self, name:str):
        """Base class for setting up custom boundary conditions

        Args:
            name (str): Name of the boundary condition
        """
        self.name = name

    def validate(self):
        assert self.computeLoss, "You need to specify a function to compute the loss"
