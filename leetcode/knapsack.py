def get_maximum_profit(
    weights,
    profits,
    capacity,
    i,
    max_profit
):
    num_items = len(weights)
    
    if i == num_items: # if gone past final item
        return max_profit 
    elif weights[i] > capacity: # if current item doesn't fit, go to the next item while keeping our current profit
        return get_maximum_profit(weights, profits, capacity, i+1, max_profit)
    else: # if current item fits, add it and continue but also traverse another path where we try without it
        return max(
            get_maximum_profit(weights, profits, capacity, i+1, max_profit),
            get_maximum_profit(weights, profits, capacity-weights[i], i+1, max_profit+profits[i])
        )
        
print(
    get_maximum_profit(
        weights=[2, 3, 1, 4],
        profits=[4, 5, 3, 7],
        capacity=5,
        i=0,
        max_profit=0
    )
)