from selectolax.parser import HTMLParser
from playwright.sync_api import sync_playwright
import requests as r
import time

def get_html(make):
    TIMEOUT = 900000
    url = f"https://www.carmax.com/cars?search={make}"
    
    with sync_playwright() as p: 
        browser = p.chromium.launch(headless=False) 
        page = browser.new_page()
        page.goto(url) 

        page.wait_for_load_state("networkidle", timeout=TIMEOUT)

        counter=1 
        total_page=5 # how many pages we would like to open. 22 items/page
        while counter<=total_page:
            try:
                print("get the button!")
                page.wait_for_selector("hzn-button",timeout=5000) # this is a shadow root
                # handle shadow root
                page.evaluate_handle('''() => {
                    const hznButton = document.querySelector('div.see-more hzn-button[variant="secondary"]')
                    const shadowRoot = hznButton.shadowRoot;
                    const button = shadowRoot.querySelector('button');
                    button.click();  // Click the button inside the shadow root
                    }''')
                counter+=1
                time.sleep(1) # stop 1s after each click to be like a human being (x

                if counter==total_page:
                    page.wait_for_selector("hzn-button",timeout=50000) # wait the whole page finishes loading
            except:
                print("No more 'see more matches' buttons found. Reach the end of the page")
                break
    

        return page.inner_html("body") 

html=get_html("honda")
# print(html)
tree=HTMLParser(html)
print(len(tree.css("a.scct--make-model-info-link")))
link_tags=tree.css("a.scct--make-model-info-link")

links=[i.attributes['href'] for i in link_tags]
print(links)