import pymysql
import pandas as pd

#-------------- A Bit of Data Process Before Upload----------
df=pd.read_csv("carinfo_after_pre_clean.csv")
print(df.head())

# process auctions: string --> boolean
df['is_auction']=df['is_auction'].map({
    'No Issue': False,
    'Auction Issue Reported':True
})

# process open_recalls: "1 open recall" --> 1
def transform_open_recalls(value):
    if value=="No Open Recalls":
        return 0
    elif value=="Not Available":
        return None
    else:
        return int(value.split()[0])

df['open_recalls']=df['open_recalls'].apply(transform_open_recalls)

# process numerica data
def convert_str_to_num(value):
    try:
        return int(value)
    except (ValueError):
        return None

df['price']=df['price'].apply(convert_str_to_num)
df['mileage']=df['mileage'].apply(convert_str_to_num)
df['year']=df['year'].apply(convert_str_to_num)
df['owners']=df['owners'].apply(convert_str_to_num)
df['cylinders']=df['cylinders'].apply(convert_str_to_num)\

# process odometer_issue_check: "No issue" --> False
def transform_odometer_issue(value):
    if value=="No Issue":
        return False
    else:
        return True

df['has_odometer_issue']=df['has_odometer_issue'].apply(transform_odometer_issue)

# proces time_to_fully_chared: 1.9 hours --> 1.9
def transform_charging_time(value):
    try:
        time=value.split(" ")[0]
        return float(time)
    except (ValueError,AttributeError):
        return None

df['time_to_fully_charge']=df['time_to_fully_charge'].apply(transform_charging_time)

# process is_electric: "non-electric" --> False
def transform_electric(value):
    if value=="non-electric":
        return False
    else:
        return True
df['is_electric']=df['is_electric'].apply(transform_electric)

# process bed_length: 4.5ft --> 4.5
def transform_bed_length(value):
    try:
        length=value.split(" ")[0]
        return float(length)
    except (ValueError,AttributeError):
        return None
df['bed_length']=df['bed_length'].apply(transform_bed_length)

# process range_when_new: 129miles --> 129
def transform_range_new(value):
    try:
        return int(value.split(" ")[0])
    except (ValueError,AttributeError) as e:
        return None
df['range_when_new']=df['range_when_new'].apply(transform_range_new)

# replace NaN in numeric columns with 0
numeric_columns = ['price', 'mileage', 'year', 'owners', 'cylinders', 'open_recalls', 'bed_length', 'time_to_fully_charge','range_when_new']
df[numeric_columns] = df[numeric_columns].fillna(0)

# replace NaN in text columns with an empty string
text_columns = ['make', 'model', 'miles_per_gallon', 'transmission', 'VIN', 'fuel', 'drive_type', 'exterior_color', 'interior_color','miles_per_gallon_equivalent']
df[text_columns] = df[text_columns].fillna('')

# Replace NaN in boolean columns with False
boolean_columns = ['accidents', 'is_electric', 'has_odometer_issue']
df[boolean_columns] = df[boolean_columns].fillna(False)
print(df.isnull().sum())  # Check for any remaining missing values

print("Processed Successfully!!!!")

# ----------------- Upload CSV to DATABASE ------------------
# Connect to the RDS database
conn = pymysql.connect(
    host='cse587carpredictor.cl26s0c6w7ut.us-east-2.rds.amazonaws.com',
    port=3306,
    user='admin',
    password='2024cse587!A',
    database='cse587carpredictor'
)

cursor=conn.cursor()
print("Connected successfully!")

insert_query = """
INSERT IGNORE INTO used_cars (
    year, make, model, price, mileage, miles_per_gallon,transmission,owners,VIN,class,is_auction,
    accidents,open_recalls,has_odometer_issue,certification,cylinders,fuel,drive_type,
    miles_per_gallon_equivalent,range_when_new,time_to_fully_charge,is_electric,bed_length,
    exterior_color,interior_color
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,%s,%s,%s,%s)
"""

data_tuples = [tuple(x) for x in df.to_numpy()]
print(len(data_tuples[0]))
print(len(insert_query.split('%s')) - 1) 
cursor.executemany(insert_query, data_tuples)
conn.commit()
cursor.close()
conn.close()
