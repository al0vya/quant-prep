v = [1,5,3,8]
w = [2,1,5,3]
C = 6
n = len(v)

memo = [[None] * (C+1) for i in range(n)] # n x (C+1) matrix; need to hold 0 - 6 for the C state which is 7 states, not 6
def knapsack_recursive(i, C):
    if i == n:
        return 0
        
    if memo[i][C]:
        return memo[i][C]
    
    inc = 0 if C-w[i] < 0 else v[i] + knapsack_recursive(i+1, C-w[i])
    exc = knapsack_recursive(i+1, C)
    
    memo[i][C] = max(inc, exc)
    return memo[i][C]

print(knapsack_recursive(0, C))

def knapsack_iterative():
    # (n+1) x (C+1) state space table
    # n+1 rows to hold the (n+1)th base case of zero
    # need to hold 0 - 6 for the C state which is 7 states, not 6
    table = [[0] * (C+1) for i in range(n+1)] 
    for i in range(n-1,-1,-1):
        for c in range(C+1):
            inc = 0 if c-w[i] < 0 else v[i] + table[i+1][c-w[i]]
            exc = table[i+1][c]
            table[i][c] = max(inc, exc)
    return table[0][C]
    
print(knapsack_iterative())