def factorial_recursive(n):
    if n <= 1:
        return 1
        
    return n * factorial_recursive(n-1)

# O(n * n)
def factorial_iterative(n):
    result = 1
    for i in range(2,n+1):
        result = multiply_add(result, i)
    
    return result

def multiply_add(a,b):
    if a > b:
        a, b = b, a
        
    result = 0
    for i in range(a):
        result += b
        
    return result

def multiply_bit(a,b):
    result = 0
    while b:
        if b & 1: # if left most bit of b is 1
            result += a
        a <<= 1
        b >>= 1
    return result

for i in range(1,100):
    for j in range(1,100):
        assert multiply_add(i,j) == i * j, f'Multiply add function incorrect at i = {i}, j = {j}'

for i in range(1,100):
    for j in range(1,100):
        assert multiply_bit(i,j) == i * j, f'Multiply bit function incorrect at i = {i}, j = {j}'

for i in range(1,10+1):
    assert factorial_iterative(i) == factorial_recursive(i), f'Factorial not equivalent for n = {i}'
    
print('Success!')