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
        while counter<=5:
            try:
                print("get the button!")
                page.wait_for_selector("hzn-button",timeout=5000)
                page.evaluate_handle('''() => {
                    const hznButton = document.querySelector('div.see-more hzn-button[variant="secondary"]')
                    const shadowRoot = hznButton.shadowRoot;
                    const button = shadowRoot.querySelector('button');
                    button.click();  // Click the button inside the shadow root
                    }''')
                counter+=1
                time.sleep(1)

                if counter==5:
                    page.wait_for_selector("hzn-button",timeout=5000)
            except:
                print("No more 'see more matches' buttons found. Reach the end of the page")
                break
    

        return page.inner_html("body") 

html=get_html("honda")
# print(html)
tree=HTMLParser(html)
print(len(tree.css("a.scct--make-model-info-link")))
tree.css("div")