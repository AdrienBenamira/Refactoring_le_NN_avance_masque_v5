import numpy as np
e = 5
for i in range(251):
    a = i
    b = i+e
    y = a^b
    alpha = int(np.log2(y)) + 1
    assert alpha == len(bin(y)[2:])
    print(bin(a), a)
    print(bin(b), b)
    print(bin(y), y, alpha)
    print(y-a, y^a)
    print("-" * 10)