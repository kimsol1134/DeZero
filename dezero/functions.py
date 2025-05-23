import numpy as np
from dezero.core import Function
from dezero.core import as_variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * cos(x) # backward 메서드 안의 모든 변수가 Variable 인스턴스, 그러므로 여기서 cos(x)는 DeZero의 cos함수
        return gx # backward 메서드 구현시 모든 계산은 반드시 DeZero 함수 사용
def sin(x):
    return Sin()(x)
class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y
    def backward(self, gy):
        x, = self.inputs
        gx = gy * -sin(x)
        return gx
def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx
def tanh(x):
    return Tanh()(x)

class Reshape(Function): 
    def __init__(self, shape):
        self.shape = shape
    def forward(self, x):
        self.x_shape = x.shape # x.shape 기억해둠
        y = x.reshape(self.shape)
        return y
    def backward(self, gy):
        return reshape(gy, self.x_shape) # 역전파에서 다시 입력 형상으로 변환
def reshape(x, shape):
    if x.shape == shape:
        return as_variable(x) # Variable 인스턴스로 반환
    return Reshape(shape)(x)    

class Transpose(Function):
    def forward(self, x):
        y = np.transpose(x)
        return y
    def backward(self, gy):
        gx = transpose(gy)
        return gx
def transpose(x):
    return Transpose()(x)