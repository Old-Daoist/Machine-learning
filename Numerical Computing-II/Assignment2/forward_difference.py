
class ForwardDifference:
    def derivative(self, f, x, h):
        return (f(x + h) - f(x)) / h
