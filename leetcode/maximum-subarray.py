import time

def test(name, function):
    array = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    
    result = function(array)
    
    if result == 6:
        print(name, str(result), "passed")
    else:
        print(name, str(result), "failed")

def brute_force(array):
    max = array[0]
    sum = 0
    
    n = len(array)
    
    for k in range(n):
        for i in range(n):
            sum = 0
            for j in range(k,i+1):
                sum += array[j]
                
                max = max if max >= sum else sum
            
    return max

def kadane(array):
    max_local  = array[0]
    max_global = array[0]
    
    n = len(array)
    
    for i in range(1,n):
        max_local = max_local + array[i] if (max_local + array[i]) > array[i] else array[i]
        
        max_global = max_global if max_global > max_local else max_local
        
    return max_global

def main():
    test("brute force", brute_force)
    test("kadane", kadane)

if __name__ == "__main__":
    main()