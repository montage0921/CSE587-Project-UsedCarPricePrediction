from selectolax.parser import HTMLParser
from playwright.sync_api import sync_playwright
import requests as r
import time
import logging
import pandas as pd

class Car_Info:
    def __init__(self,keyword,pages):
        self.keyword=keyword # "honda","sedan","red+car"(please use + to concat each word)...
        self.pages=pages # 22 pcs of info per page
        self.car_data=[]
    
    # get search page for a specific keyword
    def _get_html(self):
        TIMEOUT = 900000
        url = f"https://www.carmax.com/cars?search={self.keyword}"
        
        with sync_playwright() as p: 
            browser = p.chromium.launch(headless=False) 
            page = browser.new_page()
            page.goto(url) 

            page.wait_for_load_state("networkidle", timeout=TIMEOUT)

            counter=1 
            total_page=self.pages # how many pages we would like to open. 23 items/page
            while counter<=total_page:
                try:
                    print("See More Matches!")
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
    
   
    def _get_links(self):
        # pages: # of pages we want to scrape

        html=self._get_html()
        tree=HTMLParser(html)
        link_tags=tree.css("a.scct--make-model-info-link")
        links=[i.attributes['href'] for i in link_tags if 'href' in i.attributes]
        return links

    def scrape_car_info(self):

        links=self._get_links()
        print(links)
        print(len(links)) 

        for link in links:
            url=f"https://www.carmax.com{link}"

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=False) 
                page = browser.new_page()
                page.goto(url) 
                
                try:
                    page.wait_for_load_state("networkidle", timeout=9000)
                except TimeoutError:
                    logging.error(f"{link} loading takes too long...")
                    continue

                html=page.inner_html("body")
                tree=HTMLParser(html)

                car={} # save one car's info

                # extract year, make and model
                description=tree.css_first("h1#car-header-basic-car-info").text()
                descr_list=description.split(" ") 
                year=descr_list[0]
                make=descr_list[1]
                model=' '.join(descr_list[2:])

                car["year"]=year
                car["make"]=make
                car["model"]=model

                # extract price
                price_literal=tree.css_first("span#default-price-display").text()
                price=int(price_literal.replace('$','').replace(',','').replace('*',''))
                car["price"]=price
                # extract mileage
                mileage_literal=tree.css_first("span.car-header-mileage").text()
                mileage=int(mileage_literal.split(" ")[0].replace('K',''))*1000
                car["mileage"]=mileage

                # extract motor, drive type and color
                badges=tree.css("div.tombstone-badge")
                for badge in badges:
                    sub_label=badge.css_first('div.tombstone-badge-sub')
                    if sub_label:
                        feature_name=sub_label.text()
                        feature_value=badge.text().replace(feature_name,"").strip()
                        car[feature_name]=feature_value
                
                # append car data to the data list
                self.car_data.append(car)

    def get_car_data(self):
        df=pd.DataFrame(self.car_data)
        return df

honda=Car_Info("tesla",10)
honda.scrape_car_info()
data=honda.get_car_data()
print(data)

