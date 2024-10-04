import pandas as pd
import glob
import os

class Data_Reorganization:
    source_path = "scraped_data"
    
    def __init__(self):
        self.file_names = glob.glob(os.path.join(self.source_path, "*.csv"))

    # get data1, but need to consider condition
    def combine_data(self):
        df_list = []
        for file in self.file_names:
            file_name = os.path.basename(file) 
            one_file = pd.read_csv(file)
            df_list.append(one_file)

        combine_data = pd.concat(df_list, ignore_index = True)
        df_combine_data = pd.DataFrame(combine_data)
        df_combine_data['price'] = pd.to_numeric(df_combine_data['price'], errors = 'coerce')
        df_combine_data['price'] = df_combine_data['price'].fillna(0).astype(int)
        print(df_combine_data)
        return df_combine_data

    # car names that data1 contains
    def calculate_name_contain(self):
        car_name_contain = []
        self.combine_data()
        for filename in self.file_names:
            basename = os.path.basename(filename)
            car_name = basename.split("_")[0]
            car_name_contain.append(car_name)

        # print(car_name_contain)
        return car_name_contain

    # dataset rename to match data1
    def dataset_rename(self):
        dataset = pd.read_csv("vehicles.csv")
        df_dataset = pd.DataFrame(dataset)
        df_dataset.rename(columns = {
            "odometer" : "mileage",
            "manufacturer" : "make",
            "transmission" : "Transmission",
            "paint_color" : "Color",
            "title_status" : "State Title Brand",
            "drive" : "Drive type",
            "size" : "class"
        }, inplace = True)

        # print(df_dataset.head(100))
        return df_dataset


    # dataset sampling
    def reorganize_dataset(self):
        dataset_rename = self.dataset_rename()
        name_list = self.calculate_name_contain()
        dataset_sampling = []

        for name in name_list:
            data_select = dataset_rename[dataset_rename["make"] == name]
            data_select_top200 = data_select.sample(200)
            # 'year' and 'mileage' will fill '0' if "NaN"
            data_select_top200['year'] = data_select_top200['year'].fillna(0).astype(int)
            data_select_top200['mileage'] = data_select_top200['mileage'].fillna(0).astype(int)

            dataset_sampling.append(data_select_top200)

        combined_sampling = pd.concat(dataset_sampling, ignore_index = True)
        return combined_sampling

    # merge data1 and sampling dataset
    def merge_data(self):
        dataset_sample = self.reorganize_dataset()
        data1 = self.combine_data()
        print("Data1:\n", data1.info())
        print("Dataset Sample:\n", dataset_sample.info())

        common_columns_series = [
            "year", "make", "model", "price", "mileage", "Transmission", "Color", "VIN", 
            "class", "State Title Brand", "cylinders", "fuel", "Drive type"
        ]

        dataset_sample = dataset_sample.dropna(subset = common_columns_series)
        data1 = data1.dropna(subset = common_columns_series)

        df_merge = pd.merge(data1, dataset_sample, on = ["year"], how = 'inner')
        print("##################### Merge on fewer columns:")
        print(df_merge)
        print(df_merge.info())

        df_merge_full = pd.merge(data1, dataset_sample, on = common_columns_series, how = 'inner')
        print("##################### Full Merge:")
        print(df_merge_full)


'''
1. Enter CSE587-Project-UsedCarPricePrediction and put vehicles.csv into this path
2. Merge_data is combine by scrapping data and dataset sampling.
'''
ans = Data_Reorganization()
# scrapping data
scrpping_data = ans.combine_data()
# scrapping data + dataset sampling
# ans.merge_data()
# dataset_samling = ans.dataset_rename()

# Define the folder path
folder_path = r"/Users/jiabaoyao/Study Abroad/Projects/Data Intensive Computing/proj_phrase_1/CSE587-Project-UsedCarPricePrediction"

if not os.path.exists(folder_path):
    os.makedirs(folder_path)
df1 = pd.DataFrame(scrpping_data)
# df2 = pd.DataFrame(dataset_samling)
# Save the CSV file to the folder
file_path1 = os.path.join(folder_path, f'scrapping_data.csv')
# file_path2 = os.path.join(folder_path, f'dataset_samling.csv')
df1.to_csv(file_path1, index = False)
# df2.to_csv(file_path2, index = False)
