import heapq
top_similarities = []
num = [0,4,1,23,45,23,5,1,23,4]
for j in num:
    if len(top_similarities) < 3:
        heapq.heappush(top_similarities, j)
    else:
        if j > top_similarities[0]:
            heapq.heappop(top_similarities)  
            heapq.heappush(top_similarities, j)
print(top_similarities)