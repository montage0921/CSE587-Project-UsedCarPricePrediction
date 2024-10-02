from selectolax.parser import HTMLParser
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
import os
import asyncio
import time

class Car_Info:
    def __init__(self, keyword, pages, max_concurrency):
        self.keyword = keyword  # "honda","sedan","red+car"(please use + to concat each word)...
        self.pages = pages  # total cars=pages x 22
        self.max_concurrency = max_concurrency # number of concurrency tasks （<=4 i guess...)
        self.car_data = [] # store all extracted used car information
        self.data_processed_counter=0 # how many data processed so far

    # Go to search page and get required pages
    async def _get_html(self):
        TIMEOUT = 900000
        url = f"https://www.carmax.com/cars?search={self.keyword}"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(url)

            await page.wait_for_load_state("networkidle", timeout=TIMEOUT)

            counter = 1
            total_page = self.pages
            # handling pagination
            while counter <= total_page:
                try: 
                    print("See More Matches!")
                    # this is the "see more " button and it is a shadow root
                    await page.wait_for_selector("hzn-button", timeout=5000)  
                    # handle shadow root
                    await page.evaluate_handle('''() => {
                        const hznButton = document.querySelector('div.see-more hzn-button[variant="secondary"]')
                        const shadowRoot = hznButton.shadowRoot;
                        const button = shadowRoot.querySelector('button');
                        button.click();  // Click the button inside the shadow root
                    }''')
                    counter += 1
                    await asyncio.sleep(0.5)  # stop 0.5s after each click to mimic human behavior

                    if counter == total_page:
                        await page.wait_for_selector("hzn-button", timeout=50000)  # wait for the page to finish loading

                except Exception:
                    print("No more 'see more matches' buttons found. Reached the end of the page")
                    break

            return await page.inner_html("body")

    # Extract links for all individual car displayed on the search page
    async def _get_links(self):
        html = await self._get_html()
        tree = HTMLParser(html)
        link_tags = tree.css("a.scct--make-model-info-link")
        links = [i.attributes['href'] for i in link_tags if 'href' in i.attributes]
        return links

    # Extract all necessary features in a car's info page
    # year, make, model, price(USD), mileage, cylinders, fuel types,Miles per gallon, drive types, transmission, color, owners,condition (damage or not)
    async def _extract_car_info(self, car_page, html, conditions,stock_number):
        tree = HTMLParser(html)

        car = {}  # save one car's info

        # Extract year, make, and model
        description = tree.css_first("h1#car-header-basic-car-info").text()
        descr_list = description.split(" ")
        year = descr_list[0]
        make = descr_list[1]
        model = ' '.join(descr_list[2:])

        car["year"] = year
        car["make"] = make
        car["model"] = model

        # Extract price
        price_literal = tree.css_first("span#default-price-display").text()
        if "unavailable" not in price_literal:
            price = int(price_literal.replace('$', '').replace(',', '').replace('*', ''))
            car["price"] = price
        else:
            car["price"]="Price unavailable"

        # Extract mileage
        mileage_literal = tree.css_first("span.car-header-mileage").text()
        mileage = int(mileage_literal.split(" ")[0].replace('K', '')) * 1000
        car["mileage"] = mileage

        # Extract motor, drive type, cylinders, fuel, color etc.
        badges = tree.css("div.tombstone-badge")
        for badge in badges:
            sub_label = badge.css_first('div.tombstone-badge-sub')
            if sub_label:
                feature_name = sub_label.text()
                if "Engine" in feature_name:
                    engine_value = badge.text().strip()
                    engine_value_list = engine_value.split(", ")
                    if len(engine_value_list) >= 2:
                        car["cylinders"] = engine_value_list[0]
                        car["fuel"] = engine_value_list[1]
                    continue
                feature_value = badge.text().replace(feature_name, "").strip()
                car[feature_name] = feature_value

        # condition
        new_conditions=""
        for i in conditions:
            if i=='N':
                new_conditions+=","
            new_conditions+=i
        condition_array=new_conditions.split(",")
        car["owner"] = condition_array[0] if len(condition_array) > 0 else "N/A"
        car["frame_damage"] = condition_array[1] if len(condition_array) > 1 else "N/A"
        car["Odometer_problem"] = condition_array[2] if len(condition_array) > 2 else "N/A"

        report_url=f"https://www.carmax.com/car/{stock_number}/vehicle-history"

        await car_page.close() # close car's individual page after extract all necessary infomation

        async with async_playwright() as p:
            try:
                browser=await p.chromium.launch(headless=False)
                page=await browser.new_page()
                await page.goto(report_url)
                await page.wait_for_load_state("networkidle", timeout=8000)
                html=await page.inner_html("body")

                tree=HTMLParser(html)

                # extract VIN
                VIN=tree.css("div.decode-box-row.row.odd")[0].css("div")[2].text()
                car["VIN"]=VIN

                # extract car class: midsize, compact, CUV(crossover utility vehicle)...
                car_class=tree.css("div.decode-box-row.row")[1].css("div")[2].text()
                car["class"]=car_class

                # extract vehicy history
                history=tree.css("div#at-glance div.section-data div.col-12.col-md-6.col-lg-4")
                
                # extract state title brand, auction brand, accident/damage, open recall check, insurance loss
                # odometer check, certified pre-owned, service/repair
                # Electric Car Only: Miles per gallon equivalent (MPGe), range, time to fully charge
                for info in history:
                    feature=info.css_first("p.large-title").text()
                    value=info.css_first("span.card-footer-text-adjustment span").text()
                    car[feature]=value

            except:
                return
        self.data_processed_counter+=1
        print("number of car extracted:", self.data_processed_counter)
        return car

    # Scrape one single car's info page
    # semaphore provided by playwright limits the number of concurrency tasks
    async def scrape_page(self, url, semaphore, browser):
        async with semaphore:
            try:
                page = await browser.new_page()
                await page.goto(url)
                await page.evaluate("window.scrollBy(0, 3200);")  # solve lazy loading
                
                await page.wait_for_selector("div.history-hightlights-columns", timeout=8000)
            
                html = await page.inner_html("body")

                # Extract conditions (Handling Shadow DOM)
                # Conditions include # of past owner, frame damage or not and odometer problem
                # conditions info is contained in the shadowroot so page.evaluate() is needed to handle them
                conditions = await page.evaluate('''() => {
                    const shadowRoots = document.querySelectorAll('div.history-hightlights-columns hzn-stack');
                    if (shadowRoots.length > 0) {
                        return shadowRoots[0].textContent;
                    }
                    return "";
                }''')
               
                # extract stock number
                # used to check AutoCheck Report of each car: https://www.carmax.com/car/<stock number>/vehicle-history
                stock_number=await page.evaluate('''()=>{
                        const shadowRoots=document.querySelectorAll("div#stock-and-vin span.stock-and-vin-text")
                        if (shadowRoots.length>0){
                            return shadowRoots[1].textContent  
                        }
                        return ""                          
                    }
                ''')
                

                car = await self._extract_car_info(page,html, conditions,stock_number)
                if car is not None:
                    self.car_data.append(car)
                    print("Data extracted from:", url)
                else:
                    print(f"Failed to extract data from {url}")
                
                # await page.close()
                

            except PlaywrightTimeoutError:
                print(f"{url} loading takes too long... Skipping.")

    # Scrape all links
    async def scrape(self):
        links = await self._get_links()
        semaphore = asyncio.Semaphore(self.max_concurrency)  # Set up semaphore for concurrency

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            
            tasks = [self.scrape_page(f"https://www.carmax.com{link}", semaphore, browser) for link in links]
            await asyncio.gather(*tasks)  # Run tasks with concurrency control

            await browser.close()

    def get_car_data(self):
        df = pd.DataFrame(self.car_data)

        # Define the folder path
        folder_path = r"D:\Courses\Data Intensive Computing\Proj1\CSE587-Project-UsedCarPricePrediction\scraped_data"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Add a time stamp in CSV's file name
        # Save the CSV file to the folder
        file_path = os.path.join(folder_path, f'{self.keyword}_{timestamp}.csv')
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return df

async def main():
    # read car_brands
    with open('manufacturer_names.txt', 'r') as file:
        car_brands = [line.strip() for line in file]
    print(car_brands)


    for car in car_brands:
        car_scraper = Car_Info(keyword=car, pages=30, max_concurrency=4)
        await car_scraper.scrape()
        car_scraper.get_car_data()


asyncio.run(main())
