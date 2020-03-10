import requests
from bs4 import BeautifulSoup

#get the content from URL given
getHTML= requests.get('https://catalog.umkc.edu/course-offerings/graduate/comp-sci/')
#scrapping website using BeautifulSoup
getParsedHTML= BeautifulSoup(getHTML.text, "html.parser")
spans_name_desc = getParsedHTML.find_all(True, {'class':['title', 'courseblockdesc']})

for span in spans_name_desc:
    a=span.text
    print(a)

