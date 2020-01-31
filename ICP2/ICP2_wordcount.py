t=open("wordcount.txt",'r')
#print(t)
word_dict=dict()
for l in t:
    l=l.strip()
    l=l.lower()
    words=l.split(" ")
    for word in words:
        if word in word_dict:
            word_dict[word]=word_dict[word]+1
        else:
            word_dict[word]=1

for k in word_dict.keys():
    print(k," : ",word_dict[k])

f=open('wordcount_out.txt', 'w')
f.write(str(word_dict))
f.close()
