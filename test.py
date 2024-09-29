from selectolax.parser import HTMLParser
from playwright.sync_api import sync_playwright
import requests as r

def get_html(make):
    TIMEOUT = 900000
    url = f"https://www.carmax.com/cars?search={make}"
    
    with sync_playwright() as p: 
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page() 
        page.goto(url) 

        page.wait_for_load_state("networkidle", timeout=TIMEOUT)
    

        return page.inner_html("body") 

html=get_html("ford")
# print(html)
tree=HTMLParser(html.json)
print(tree)