from abc import ABC # abstract class python

class Boco(ABC):
    def __init__(self, name:str):
        """Base class for setting up boundary conditions

        Args:
            name (str): name of the boundary conditions. Example: "Initial conditions"
        """
        self.name = name

    def validate(self):
        assert self.computeLoss, "You need to specify a function to compute the loss"
