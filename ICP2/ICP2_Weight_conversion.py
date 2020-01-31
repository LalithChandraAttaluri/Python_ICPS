n = input("Enter the number of stuents that require weight conversion? ")
list=[]
list_conv=[]
for i in range(int(n)):
    x=int(input("Enter the weight>> "))
    list.append(x)
print(list)

for j in list:
    #j=int(j)
    j=j*0.454
    list_conv.append(j)

print(list_conv)