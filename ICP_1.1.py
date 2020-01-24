s = input('Enter the string you wish to manipulate: ')
unwanted_char = ('p','y','d','a')
s_modified=''.join(i for i in s if not i in unwanted_char)
print(s_modified[::-1])