# step 37 텐서를 다루다.

import numpy as np
from dezero import Variable
import dezero.functions as F

# 원소별 연산을 수행하는 함수(add,sin)은 입출력 데이터가 스칼라라고 가정하고 순전파와 역전파를 구현할수 있음.
# 이 경우에 텐서를 입력해도 역전파가 올바르게 성립

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t) # 추후 구현

y.backward(retain_grad=True)
print(y.grad)
print(t.grad)
print(x.grad)
print(c.grad)

#x.shape == x.grad.shape