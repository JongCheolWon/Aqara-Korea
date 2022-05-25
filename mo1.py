def add(a, b):
    return a + b
def sub(a, b):
    return a - b

import mo1
print(mo1.add(3, 4))
print(mo1.sub(4, 2))

from mo1 import add
add(3, 4)