def find_triplet_zero_sum(array):
    array.sort()
    
    n = len(array)
    
    triplets = []
    
    for i in range(n):
        left  = i + 1
        right = n - 1
        
        if i > 0:
            # if it's the same number continue because we've already tried
            if array[i] == array[i-1]:
                continue
        
        while left < right:
            sum = array[i] + array[left] + array[right]
            
            if sum == 0:
                triplets.append((array[i], array[left], array[right]))
                left  += 1
                right -= 1
            elif sum > 0:
                right -= 1
            elif sum < 0:
                left += 1
    
    return triplets
    
def test(array, answer):
    triplets = find_triplet_zero_sum(array)
    
    if triplets == answer:
        print(triplets, "passed")
    else:
        print(triplets, "failed")
        
def main():
    test(
        array=[-1, 0, 1, 2, -1, -4],
        answer=[(-1, -1, 2), (-1, 0, 1)]
    )
    
    test(
        array=[1, -2, 1, 0, 5],
        answer=[(-2, 1, 1)]
    )
    
if __name__ == "__main__":
    main()