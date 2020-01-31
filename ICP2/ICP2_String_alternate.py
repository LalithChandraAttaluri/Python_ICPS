def string_alternative():
    s=input("Please enter the string>> ")
    for i in range(len(s)):
        if(i%2)==0:
            out1=s[i]
            print(out1,end="")
string_alternative()