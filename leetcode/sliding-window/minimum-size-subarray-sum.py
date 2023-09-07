def sliding_window(array, target):
    n = len(array)
    
    left = 0
    
    sum = 0
    
    INT_MAX = int(2 ** 63 + 1)
    
    min = INT_MAX
    
    for i in range(n):
        sum += array[i]
        
        while sum >= target:
            width = i - left + 1 # +1 to account for 0 index
            
            min = min if min < width else width
            
            sum -= array[left]
            
            left += 1
            
    return min if min != INT_MAX else 0
    
def test(array, target, answer):
    min = sliding_window(array, target)
    
    if min == answer:
        print("passed for", array)
    else:
        print("failed for", array)

def main():
    test(
        array=[2,3,1,2,4,3],
        target=7,
        answer=2
    )
    
    test(
        array=[1,4,4],
        target=1,
        answer=1
    )
    
    test(
        array=[1,1,1,1,1,1,1,1],
        target=11,
        answer=0
    )
    
    
if __name__ == "__main__":
    main()