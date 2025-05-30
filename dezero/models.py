from dezero import Layer
from dezero import utils
import dezero.functions as F
import dezero.layers as L

# Layer 클래스 상속받아 시각화 메서드 추가

class Model(Layer):
    def plot(self, *inputs, to_file='model.png'):
        y = self.forward(*inputs)
        return utils.plot_dot_graph(y, verbose=True, to_file=to_file)
# Layer 클래스 처럼 사용가능 ex) class TwoLatyerNet(Model):

class MLP(Model):
    def __init__(self, fc_output_sizes, activation=F.sigmoid):
        super().__init__()
        self.activation = activation
        self.layers = []

        for i, out_size in enumerate(fc_output_sizes):
            layer = L.Linear(out_size)
            setattr(self, 'l' + str(i), layer)
            self.layers.append(layer)

    def forward(self,x):
        for l in self.layers[:-1]:
            x = self.activation(l(x))
        return self.layers[-1](x)