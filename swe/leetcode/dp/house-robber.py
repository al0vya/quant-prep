def func_rec(arr, i=0):
    if i >= len(arr):
        return 0
    
    return max(
        func_rec(arr, i+2) + arr[i],
        func_rec(arr, i+1)
    )
    
def func_memo(arr, dp, i=0):
    if i < len(arr) and dp[i] != None:
        return dp[i]

    if i >= len(arr):
        return 0

    dp[i] = max(
        func_memo(arr, dp, i+2) + arr[i], 
        func_memo(arr, dp, i+1)
    )

    return dp[i]

nums = [183,219,57,193,94,233,202,154,65,240,97,234,100,249,186,66,90,238,168,128,177,235,50,81,185,165,217,207,88,80,112,78,135,62,228,247,211]

print(func_memo(arr=nums, dp=[None]*(len(nums))))

print(func_rec(arr=[1,2,3,1]))
print(func_rec(arr=[2,7,9,3,1]))
