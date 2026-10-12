
class BackwardDifference:
    def derivative(self, f, x, h):
        return (f(x) - f(x - h)) / h
