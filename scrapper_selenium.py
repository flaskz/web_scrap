# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:32:17 2018

@author: Lucas
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import json

url = r'https://www.zapimoveis.com.br/aluguel/apartamentos/pr+curitiba/'

browser = webdriver.Chrome('E:/webdrivers/chromedriver.exe')
browser.get(url)
print(browser.page_source)

soup = BeautifulSoup(browser.page_source, 'html.parser')
html = soup.prettify('utf-8') 

lst=[]
for item in soup.find_all('article', class_='minificha'):
    lst.append(json.loads(item['data-clickstream']))
    
try:
    element = browser.find_element_by_id('proximaPagina')
    browser.execute_script("arguments[0].click();", element)
except Exception as e:
    print(e)
element = browser.find_element_

browser.quit()