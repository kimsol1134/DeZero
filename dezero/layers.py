from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F
import os
class Layer:
    def __init__(self):
        self._params = set()

    def __setattr__(self, name, value): # 인스턴스 변수 설정할때 호출되는 특수 메서드, 이름이 name인 인스턴스 변수에 값으로 value를 전달
        if isinstance(value, (Parameter, Layer)): # value가 Parameter 인스턴스라면, Layer도 추가
            self._params.add(name) # _params 인스턴스 변수에 매개변수 보관
        super().__setattr__(name, value)

    def __call__(self, *inputs):
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,) # 튜플로 변환
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs] # 입출력 약한참조
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()

    def params(self):
        for name in self._params: #layer 인스턴스에 담긴 parameter 인스턴스 꺼내줌
            obj = self.__dict__[name]

            if isinstance(obj, Layer): # Layer에서 매개변수 꺼내기
                yield from obj.params() # Layer면 재귀적으로 obj.params호출
            else :
                yield obj

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    def cleargrads(self): #cleargrad(S) layer가 가진 '모든' 매개변수에 대해 claergard 호출
        for param in self.params():
            param.cleargrad()


    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key) # 재귀적 호출
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items()
                      if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]

class Linear(Layer):
    def __init__(self, out_size, nobias=False, dtype=np.float32, in_size=None): # in_size 지정안함
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name='W')
        if self.in_size is not None: # in_size가 지정되어 있지 않다면 가중치 생성시점 forward로 늦춤
            self._init_W()

        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')

    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I, O).astype(self.dtype) * np.sqrt(1 / I)
        self.W.data = W_data

    def forward(self, x):
        # 데이터 흘러보내는 시저멩 가중치 초기화 
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y