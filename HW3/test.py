import heapq



number = [10,2,3,4,52,31,4,5,6]
top_similarities = []
for i in number:
    if len(top_similarities) < 3:
        heapq.heappush(top_similarities,i)
    else:
        if i > top_similarities[0]:
            heapq.heappop(top_similarities)  
            heapq.heappush(top_similarities, i)
print(top_similarities)