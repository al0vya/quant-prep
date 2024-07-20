import time

def fibonacci_recursive(n):
    if n == 1 or n == 2:
        return 1
        
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_memoized(n, memo):
    if memo[n] != None:
        return memo[n]
    
    if n == 1 or n == 2:
        memo[n] = 1
        
        return 1
    
    fibonacci_n_sub_1 = fibonacci_memoized(n-1, memo)
    
    fibonacci_n_sub_2 = fibonacci_memoized(n-2, memo)
    
    memo[n-1] = fibonacci_n_sub_1
    memo[n-2] = fibonacci_n_sub_2

    return fibonacci_n_sub_1 + fibonacci_n_sub_2
    
def fibonacci_bottom_up(n):
    memo = [None] * n
    
    memo[0] = 1
    memo[1] = 1
    
    for i in range(2,n):
        memo[i] = memo[i-1] + memo[i-2]
        
    return memo[n-1]

def main():
    n = 38
    
    start = time.time()
    print( "Recursive:", str( fibonacci_recursive(n) ) )
    end   = time.time()
    print( "Elapsed: ", str(end - start) )
    
    start = time.time()
    print( "Bottom up:", str( fibonacci_bottom_up(n) ) )
    end   = time.time()
    print( "Elapsed: ", str(end - start) )
    
    start = time.time()
    print( "Memoized: ", str( fibonacci_memoized(n=n, memo=[None] * (n+1)) ) )
    end   = time.time()
    print( "Elapsed: ", str(end - start) )

if __name__ == "__main__":
    main()