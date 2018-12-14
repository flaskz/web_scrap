# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 19:32:17 2018

@author: Lucas
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import json
import time

url = r'https://www.zapimoveis.com.br/aluguel/apartamentos/pr+curitiba/'

browser = webdriver.Chrome('C:/webdrivers/chromedriver.exe')
browser.get(url)
# print(browser.page_source)

lst=[]
pagina_atual = 0
dupes = []
for i in range(1000000):
    
    soup = BeautifulSoup(browser.page_source, 'html.parser')
    html = soup.prettify('utf-8')
    
    while pagina_atual == soup.find('div', {'class':'pagination'}).find('input')['value']:
        soup = BeautifulSoup(browser.page_source, 'html.parser')
        html = soup.prettify('utf-8') 
        print('waiting load...')
        time.sleep(1)
    
    pagina_atual = soup.find('div', {'class':'pagination'}).find('input')['value']
    print('pagina atual: ', pagina_atual)
    
    for item in soup.find_all('article', class_='minificha'):
        aux = json.loads(item['data-clickstream'])
        aux['url'] = json.loads(item['data-layer'])['urlAnt']
        aux['pagina'] = soup.find('div', {'class':'pagination'}).find('input')['value']
        if aux['listingId'] not in [x['listingId'] for x in lst]:
            lst.append(aux)
        else:    
            print('Aonde:', aux['pagina'], aux['listingId'])
            print('repetido')
            dupes.append((aux, [x for x in lst if aux['listingId'] == x['listingId']][0]))
    
            
    try:
        element = browser.find_element_by_id('proximaPagina')
        try:
            if element.get_attribute('disabled'):
                print('disabled')
                break
            else:
                print('clicando proxima pagina...')
                browser.execute_script("arguments[0].click();", element)
        except Exception as e2:
            print(e2)
    except Exception as e:
        print(e)

browser.quit()

import pickle

with open('apts.pkl', 'wb') as f:
    pickle.dump(lst, f)

#with open('apts.pkl', 'rb') as f:
#    teste = pickle.load(f)

#
#len(set([x['listingId'] for x in lst]))
#
#a = [x for x in lst if x['listingId'] == '20303408']
#