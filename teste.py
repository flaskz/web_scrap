# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 12:28:38 2018

@author: l.ikeda
"""

import requests
from bs4 import BeautifulSoup

import json

proxies = {"http"  : "http://proxy.cinq.com.br:3128",
           "https"  : "http://proxy.cinq.com.br:3128"}

url = r'https://www.zapimoveis.com.br/aluguel/apartamentos/pr+curitiba/?__zt=ad:a'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.110 Safari/537.36'}
r = requests.get(url, proxies=proxies, headers=headers)

lst = []

cnt = 0
with open('logs.txt', 'w') as logs:
    for i in range(1,3):
        try:
            payload = r'#{"precomaximo":"2147483647","parametrosautosuggest":[{"Bairro":"","Zona":"","Cidade":"CURITIBA","Agrupamento":"","Estado":"PR"}],"pagina":'+str(i)+',"ordem":"Relevancia","paginaOrigem":"ResultadoBusca","semente":"674745354","formato":"Lista"}'
            r = requests.get(url+payload, proxies=proxies, headers=headers)
            logs.write(str(r.status_code)+': '+str(r.url)+'\n')
        
            soup = BeautifulSoup(r.content, 'html.parser')
            html = soup.prettify('utf-8') 

            for item in soup.find_all('article', class_='minificha'):
                lst.append(json.loads(item['data-clickstream']))
        except Exception as e:
            logs.write(str(e)+'\n')
    
        
import pickle

with open('casas.pkl', 'wb') as f:
    pickle.dump(lst, f)

'''
with open('casas.pkl', 'rb') as f:
    teste = pickle.load(f)
'''