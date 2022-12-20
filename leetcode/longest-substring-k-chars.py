def sliding_window(string, max_char):
    n = len(string)
    
    current_chars = {}
    
    left = 0
    
    INT_MIN = -int(2 ** 63 + 1)
    
    width = INT_MIN
    
    for i in range(n):
        right_char = string[i]
        
        if right_char not in current_chars.keys():
            current_chars[right_char] = 1
        else:
            current_chars[right_char] += 1
            
        while len(current_chars.keys()) > max_char:
            left_char = string[left]
            
            current_chars[left_char] -= 1
            
            if current_chars[left_char] == 0:
                del current_chars[left_char]
            
            left += 1
            
        width = max(width, i - left + 1)
        
    return n if width == INT_MIN else width
    
def test(string, max_char, answer):
    result = sliding_window(string, max_char)
    
    if result == answer:
        print(string, "passed")
    else:
        print(string, "failed")
        
def main():
    test(
        string="ababcbcca",
        max_char=2,
        answer=5
    )
    
    test(
        string="abcde",
        max_char=1,
        answer=1
    )
    
if __name__ == "__main__":
    main()