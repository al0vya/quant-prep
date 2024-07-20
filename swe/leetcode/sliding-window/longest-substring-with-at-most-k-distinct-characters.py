def longestSubstringKchars(string, max_char):
    left, right = 0, 0
    
    chars = {}
    
    INT_MIN = -int(2 ** 31 + 1)
    
    ans = INT_MIN
    
    while right < len(string):
        char = string[right]
        
        if char not in chars:
            chars[char] = 1
        else:
            chars[char] += 1
        
        while len(chars.keys()) > max_char:
            char = string[left]
            
            chars[char] -= 1
            
            if chars[char] == 0:
                del chars[char]
                
            left += 1
            
        ans = max(ans, right - left + 1)
        
        right += 1
            
    return ans
            
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
    result = longestSubstringKchars(string, max_char)
    
    print(string, result, "passed" if result == answer else "failed")
         
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