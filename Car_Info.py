from selectolax.parser import HTMLParser
from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError
import pandas as pd
import os
import asyncio

class Car_Info:
    def __init__(self, keyword, pages, max_concurrency):
        self.keyword = keyword  # "honda","sedan","red+car"(please use + to concat each word)...
        self.pages = pages  # 22 pcs of info per page
        self.max_concurrency = max_concurrency # number of concurrency tasks
        self.car_data = [] # store all extracted used car information

    # go to search page and extract links of 
    async def _get_html(self):
        TIMEOUT = 900000
        url = f"https://www.carmax.com/cars?search={self.keyword}"

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=False)
            page = await browser.new_page()
            await page.goto(url)

            await page.wait_for_load_state("networkidle", timeout=TIMEOUT)

            counter = 1
            total_page = self.pages  # How many pages we would like to open, 23 items per page
            while counter <= total_page:
                try:
                    print("See More Matches!")
                    await page.wait_for_selector("hzn-button", timeout=5000)  # this is a shadow root
                    # handle shadow root
                    await page.evaluate_handle('''() => {
                        const hznButton = document.querySelector('div.see-more hzn-button[variant="secondary"]')
                        const shadowRoot = hznButton.shadowRoot;
                        const button = shadowRoot.querySelector('button');
                        button.click();  // Click the button inside the shadow root
                    }''')
                    counter += 1
                    await asyncio.sleep(0.5)  # stop 1s after each click to mimic human behavior

                    if counter == total_page:
                        await page.wait_for_selector("hzn-button", timeout=50000)  # wait for the page to finish loading
                except Exception:
                    print("No more 'see more matches' buttons found. Reached the end of the page")
                    break

            return await page.inner_html("body")

    async def _get_links(self):
        html = await self._get_html()
        tree = HTMLParser(html)
        link_tags = tree.css("a.scct--make-model-info-link")
        links = [i.attributes['href'] for i in link_tags if 'href' in i.attributes]
        return links

    async def _extract_car_info(self, html, page):
        tree = HTMLParser(html)

        car = {}  # Save one car's info

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
        price = int(price_literal.replace('$', '').replace(',', '').replace('*', ''))
        car["price"] = price

        # Extract mileage
        mileage_literal = tree.css_first("span.car-header-mileage").text()
        mileage = int(mileage_literal.split(" ")[0].replace('K', '')) * 1000
        car["mileage"] = mileage

        # Extract motor, drive type, and color
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

        # Extract conditions (Handling Shadow DOM)
        conditons = await page.evaluate('''() => {
            const shadowRoots = document.querySelectorAll('div.history-hightlights-columns hzn-stack');
            if (shadowRoots.length > 0) {
                return shadowRoots[0].textContent;
            }
            return "";
        }''')

        new_condition = ",".join([i for i in conditons if i == 'N']).split(",")

        car["owner"] = new_condition[0] if len(new_condition) > 0 else "N/A"
        car["frame_damage"] = new_condition[1] if len(new_condition) > 1 else "N/A"
        car["Odometer_problem"] = new_condition[2] if len(new_condition) > 2 else "N/A"

        return car

    async def scrape_page(self, url, semaphore, browser):
        async with semaphore:
            try:
                page = await browser.new_page()
                await page.goto(url)
                await page.evaluate("window.scrollBy(0, 3200);")  # solve lazy loading

                await page.wait_for_selector("div.history-hightlights-columns", timeout=9000)
                html = await page.inner_html("body")

                car = await self._extract_car_info(html, page)
                self.car_data.append(car)
                print("Data extracted from:", url)
                await page.close()
            except PlaywrightTimeoutError:
                print(f"{url} loading takes too long... Skipping.")

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
        folder_path = r"C:\Users\19692\Downloads\UB CS\2024 Fall\Homework\CES 587\CSE587-Project-UsedCarPricePrediction\scraped_data"

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        timestamp = time.strftime("%Y%m%d-%H%M%S")  # Add a time stamp in CSV's file name
        # Save the CSV file to the folder
        file_path = os.path.join(folder_path, f'{self.keyword}_{timestamp}.csv')
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
        return df

async def main():
    car_scraper = Car_Info("mazada", 1, max_concurrency=2)
    await car_scraper.scrape()
    car_scraper.get_car_data()

asyncio.run(main())
