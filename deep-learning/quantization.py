import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=0)

array = rng.normal(scale=10, size=10000)

# min/max of floats range = [a,b]; in practice, finding these is not trivial at all
a = np.min(array)
b = np.max(array)

# range of 8-bit signed int: [aq, bq] = [-128, 217]
aq = -128
bq =  127

# derived with quantization formula:
# xq = round(x/S - Z) => x = S * (xq + Z)
# then, quantizing a must give aq, while
# quantizing b must give bq
S = (b - a) / (bq - aq)

# another restriction is that when we dequantize the xq 
# corresponding to zero, we should get back exactly zero 
# that's where the rounding comes from so that
Z = int(round((b * aq - bq * a) / (b - a), 0))

# if-else is for clipping: quantized values outside range
# [aq, bq] get clipped to aq (too small) or bq (too big)
def quantize(x):
    # THIS WOULD BE A CUDA KERNEL WHERE WE READ FLOAT32 BUT STORE INT8
    xq = int(round(x/S + Z, 0))
    
    # clipping
    if xq > bq: return bq
    if xq < aq: return aq
    
    return xq
    
def dequantize(xq):
    # THIS WOULD BE A CUDA KERNEL THAT VECTOR-READS SEVERAL INT8 BUT THEN DOES A TYPE
    # CAST TO PREVENT OVERFLOW WHEN COMPUTING (AND THEN STORING) XQ - Z IN INT32/FLOAT32
    return S * (xq - Z) 
    
def test(test_bool, test_str):
    if test_bool:
        print(f'Test success: {test_str}')
    else:
        print(f'Test failure: {test_str}')

# test 1
test(quantize(a) == aq, 'quantizing a should give aq')
    
# test 2
test(quantize(b) == bq, 'quantizing b should give bq')
    
# test 3
test(dequantize(quantize(0)) == 0, 'quantizing 0 then dequantizing it should give back exactly 0')

# test 4
test(quantize(a-10) == aq, 'quantizing a number smaller than a should clip to aq')
    
# test 5
test(quantize(b+10) == bq, 'quantizing a number bigger than b should clip to bq')

array_q  = np.array([quantize(x)   for x in array])
array_dq = np.array([dequantize(x) for x in array_q])

# test 6
test(np.min(array_q) == aq, 'minimum quantized value in array should be equal to aq')

# test 7
test(np.max(array_q) == bq, 'maximum quantized value in array should be equal to bq')

fig, axs = plt.subplots(nrows=3)

axs[0].hist(array,    bins=256); axs[0].set_title('Raw data')
axs[1].hist(array_q,  bins=256); axs[1].set_title('Quantized')
axs[2].hist(array_dq, bins=256); axs[2].set_title('Dequantized')

fig.tight_layout()
plt.show()

