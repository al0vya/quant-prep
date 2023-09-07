'''
Meeting rooms

Problem statement:

Given an array of meeting time intervals consisting of start and end times
[[s1,e1],[s2,e2],...], (si < ei), determine if a person could attend all meetings.
'''

def meetingRooms(intervals):
    START, END = 0, 1
    
    intervals.sort(key=lambda x : x[START])
    
    for i, interval in enumerate(intervals[1:]):
        if interval[START] < intervals[i][END]:
            return False
            
    return True

def test(intervals, ans):
    output = meetingRooms(intervals)
    
    if output == ans:
        print("Passed")
    else:
        print("Failed")    
    
def main():
    test(
        intervals=[(0,30),(5,10),(15,20)],
        ans=False
    )
    
    test(
        intervals=[(5,8),(9,15)],
        ans=True
    )
    
if __name__ == "__main__":
    main()