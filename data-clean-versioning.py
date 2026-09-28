import pandas as pd
import boto3
from datetime import date

# load raw csv from local resource
path=r"C:\Users\Akashdeep Singh\Desktop\Mlops_house_predication_raw_data.csv"
data=pd.read_csv(path)
df=pd.DataFrame(data)

print("=============Before cleaning==============")
print(df.isnull().sum())
print(f"shape before : {df.shape}")
print()

df_clean= df.dropna() # data cleaning
print("============after cleaning===========")
print(df_clean.isnull().sum())
print(f"shape After : {df_clean.shape}")

# saved cleaned CSV locally
clean_path=r"C:\Users\Akashdeep Singh\Desktop\Mlops_house_predication_clean_v1.csv"
df_clean.to_csv(clean_path,index=False)

# upload to s3 as proccessed version

s3=boto3.client('s3')
BUCKET="akash-7777"
def upload_proccessed_data(local_path):
    key=f"proccessed/{date.today()}/Mlops_house_predication_clean_v1.csv"
    s3.upload_file(local_path,BUCKET,key)
    print(f"\nUploaded to s3://{BUCKET}/{key}")
    return key

upload_proccessed_data(clean_path)