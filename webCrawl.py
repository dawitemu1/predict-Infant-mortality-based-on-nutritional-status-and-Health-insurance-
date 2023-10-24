'''
Author: <Johnny Hsu, aka. Yu Wei, Hsu>
Date: <10/09/2016>
Version: 0.2 
Description: Web Crawling 
'''
from collections import deque
import requests
from urllib import parse
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(usage='webCrawl.py url [-n N [N ...]] [-p P [P ...]] [-k K [K ...]] [-e E [E ...]] [-f F [F ...]]')
parser.add_argument("url", type = str, help = "assign url to start crawl")
parser.add_argument("-n", type = int, help = "assign maximum number of web page to crawl, default is 1000")
parser.add_argument("-p", type = str, help = "assign prefix to filter url, such as http://www.trulia.com/")
parser.add_argument("-k", type = str, help = "assign key word to filter url, such as Tucson")
parser.add_argument("-e", type = str, help = "assign extension to filter url, such as .pdf, .php, .asp, .html")
parser.add_argument("-f", type = str, help = "assign file as output ")

args = parser.parse_args()



class Spider:
    def __init__(self):
        self.base_url = ""
        self.url = []


    def get_link(self, url):
        self.base_url = url

        if not url:
            raise RuntimeError('url must be specified.')
        else:
            soup = BeautifulSoup(requests.get(url).content, "lxml")

        html_tag = "a"
        tag_attribute = "href"

        try:
            links = soup.find_all(html_tag)

            for link in links:
                relative_link = link[tag_attribute]
                self.url.append(parse.urljoin(self.base_url, relative_link))
            return self.url

        except:
            raise RuntimeError('No link found')


def crawling(url, max_pages, link_prefix, key_word, link_extension, file):  

    pages_to_visit = deque([url])
    number_visited = 0
    pages_visited = set([url])

    while number_visited < max_pages and pages_to_visit:
     
        url = pages_to_visit.popleft() # Queue , use deque like queue.
        number_visited += 1
        try:
            spider = Spider()
            links = spider.get_link(url)
           
            for link in links:
                if link not in page_visited:
                    pages_to_visit.append(link)
                    pages_visited.add(link)
        except:
            pass

    # pages_visited = list(set(pages_visited))
    with open(file, "w") as fp:
         for link in pages_visited:
            if  link_prefix in link and link_extension in link and key_word in link:
                fp.write(link + "\n")
   

def main():
    if "http" in args.url:
        url = args.url
    else:
        url = "http://" + args.url
    
    max_pages = args.n
    link_prefix = args.p
    key_word = args.k
    link_extension = args.e
    file = args.f

    if not args.n: 
        max_pages = 1000
    if not args.p:
        link_prefix = ""
    if not args.k:
        key_word = "" 
    if not args.e:
        link_extension = ""  
    if not args.f:
        file = "links.txt"

    crawling(url, max_pages, link_prefix, key_word, link_extension, file)

if __name__ == '__main__':
    main()


    
