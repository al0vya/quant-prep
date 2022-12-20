def compute_sorted_squared_array(array):
    n = len(array)
    
    left  = 0
    right = n - 1
    
    sorted_squared_array = [None] * n
    
    for i in range(n-1,-1,-1):
        if abs(array[left]) >= abs(array[right]):
            sorted_squared_array[i] = array[left] * array[left]
            left += 1
        else:
            sorted_squared_array[i] = array[right] * array[right]
            right -= 1
            
    return sorted_squared_array

array = [-4,-1,0,3,10]
print(array)

sorted_squared_array = compute_sorted_squared_array(array)
print(sorted_squared_array)