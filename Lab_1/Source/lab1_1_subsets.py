def sub_lists(list1):

    # Create a list to store all the possible subsets
    sublist = []

    # first loop
    for i in range(len(list1)):
        # second loop
        for j in range(i + 1, len(list1)+1):
            # slice the subarray
            sub = list1[i:j]
            if sub not in sublist:
                sublist.append(sub)
    sublist.sort(key = len)
    return sublist

#l1 = [1,2,2,3]
n = int(input("Enter the size of list : "))
l1 = list(int(num) for num in input("Enter the list numbers separated by space: ").strip().split())[:n]
print("New List: ", l1)
print("All the possible subsets of the given list is: ",sub_lists(l1))