'''
Meeting rooms II

Problem statement:

Given an array of meeting time intervals consisting of start and end times
[[s1,e1],[s2,e2],...], (si < ei), find the minimum number of meeting rooms required.
'''

def meetingRooms(intervals):
    START, END = 0, 1
    
    starts, ends = [], []
    
    for interval in intervals:
        starts.append(interval[START])
        ends.append(interval[END])
    
    starts.sort()
    ends.sort()
    
    s, e = 0, 0
    
    ans = 0
    curr = 0
    
    # almost like a two pointer problem
    # when a meeting starts, add to the count
    # when a meeting ends, take away from the count
    # the minimum between start and end dictates
    # which event (start or end) occurs;
    # the event that occurs' pointer is incremented
    while s < len(starts):
        if starts[s] < ends[e]:
            curr += 1
            s += 1
        else:
            curr -= 1
            e += 1
            
        ans = max(ans, curr)
            
    return ans
    

def test(intervals, ans):
    output = meetingRooms(intervals)
    
    if output == ans:
        print("Passed")
    else:
        print("Failed")    
    
def main():
    test(
        intervals=[(0,30),(5,10),(15,20)],
        ans=2
    )
    
    test(
        intervals=[(2,7)],
        ans=1
    )
    
    test(
        intervals=[(5,8),(9,15)],
        ans=1
    )
    
if __name__ == "__main__":
    main()