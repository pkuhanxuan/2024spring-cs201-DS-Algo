from collections import deque

def findthemultiple(n):
    d = deque()
    d.append(1)
    while d:
        temp = d.popleft()
        d.append(temp*10)
        d.append(temp*10+1)
        if temp%n == 0:
            return temp

while True:
    n = int(input())
    if n==0:
        break
    print(findthemultiple(n))