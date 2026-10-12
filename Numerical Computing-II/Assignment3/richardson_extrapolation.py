from central_difference import CentralDifference

class RichardsonExtrapolation:
    def __init__(self):
        self.central = CentralDifference()

    def derivative(self, f, x, h):
        d_h = self.central.derivative(f, x, h)
        d_half = self.central.derivative(f, x, h / 2)

        return (4 * d_half - d_h) / 3