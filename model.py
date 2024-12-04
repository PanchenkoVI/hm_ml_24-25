from fastapi import File, UploadFile, HTTPException, FastAPI
from sklearn.preprocessing import OneHotEncoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
from joblib import load
import pandas as pd
import numpy as np
import random
import io

random.seed(42)
np.random.seed(42)

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
    max_torque_rpm: str

class Items(BaseModel):
    objects: List[Item]

def extract_values(row):
    fv = row[0].replace('@', '').lower().replace('nm', '').replace('kgm', '').replace('nan', '').replace(
        '380(38.7)', '380').replace('110(11.2)', '110')
    first_value = float(fv) if fv != '' else None
    result_max_v = -1
    for lst in row:
        if lst == row[0]:
            continue
        max_value = lst.replace('@', '').lower().replace('at', '').replace('nm', '').replace('kgm', '').replace('/','').replace(
            'nan', '').replace('110(11.2)', '110').replace('380(38.7)', '380').replace('rpm', '').replace('~', '-')
        if max_value == '(' or max_value == ')' or max_value == '':
            continue
        max_value = float(max(max_value.split('-')).replace('(', '').replace(')', ''))
        if max_value and max_value > result_max_v:
            result_max_v = max_value
    if result_max_v == -1:
        result_max_v = None
    return first_value, result_max_v

def preprocess_data(df):
    df2 = df.drop('selling_price', axis=1)
    lst_ind_not_dupl = df2.drop_duplicates().reset_index()['index'].tolist()
    df = df2[df2['index'].isin(lst_ind_not_dupl)].drop('index', axis=1).reset_index(drop=True)
    df['mileage'] = df[~df['mileage'].isna()]['mileage'].apply(lambda x: float(str(x).split()[0]))
    df['engine'] = df[~df['engine'].isna()]['engine'].apply(lambda x: float(str(x).split()[0]))
    df['max_power'] = df[~df['max_power'].isna()]['max_power'].apply(lambda x: float(str(x).replace('bhp', '').strip()) if str(x).replace('bhp', '').strip() != '' else None)
    df['torque'] = df['torque'].apply(lambda row: str(row).replace(',', '.').split(' '))
    df[['torque', 'max_torque_rpm']] = df['torque'].apply(extract_values).apply(pd.Series)
    df = df.fillna(df.median(numeric_only=True))
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    df['Owners'] = (df['owner'].isin(['Third Owner', 'Fourth & Above Owner'])).astype(int)
    df['Dealer12'] = ((df['owner'].isin(['First Owner', 'Second Owner'])) &
                               (df['seller_type'] == 'Dealer')).astype(int)
    df['Individual12'] = ((df['owner'].isin(['First Owner', 'Second Owner'])) &
                                   (df['seller_type'] == 'Individual')).astype(int)
    df['km_driven_log'] = np.log1p(df['km_driven'])
    df['name'] = df['name'].str.split().str[0]
    X_train_cat = df.copy()
    lst_num_columns = ['km_driven_log', 'Owners', 'Dealer12', 'Individual12', 'year', 'km_driven', 'mileage', 'engine',
                       'max_power', 'torque', 'max_torque_rpm']
    scaler = load('scaler.pkl')
    X_train_cat_num = X_train_cat[lst_num_columns].copy()
    X_train_cat_num['test1'] = X_train_cat_num['engine'] * X_train_cat_num['max_power']
    X_train_cat_num['test2'] = X_train_cat_num['km_driven'] / X_train_cat_num['max_torque_rpm']
    X_train_scaled = scaler.transform(X_train_cat_num)
    ohe = OneHotEncoder(sparse_output=False, drop='first')
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=scaler.get_feature_names_out())
    X_train_cat_ohe = ohe.fit_transform(X_train_cat[['seats', 'name', 'fuel', 'seller_type', 'transmission', 'owner']])
    X_train_cat_ohe =  pd.DataFrame(X_train_cat_ohe, columns=ohe.get_feature_names_out())
    X_train_cat_ohe = pd.concat([X_train_scaled, X_train_cat_ohe], axis=1)
    columns_models=['Dealer12', 'Individual12', 'Owners', 'engine', 'fuel_CNG',
     'fuel_Diesel', 'fuel_LPG', 'km_driven', 'km_driven_log', 'max_power',
     'max_torque_rpm', 'mileage', 'name_Ambassador', 'name_Audi', 'name_BMW',
     'name_Chevrolet', 'name_Daewoo', 'name_Datsun', 'name_Fiat',
     'name_Force', 'name_Ford', 'name_Honda', 'name_Hyundai', 'name_Isuzu',
     'name_Kia', 'name_MG', 'name_Mahindra', 'name_Maruti',
     'name_Mercedes-Benz', 'name_Mitsubishi', 'name_Nissan', 'name_Peugeot',
     'name_Renault', 'name_Skoda', 'name_Tata', 'name_Toyota',
     'owner_First Owner', 'owner_Fourth & Above Owner',
     'owner_Test Drive Car', 'owner_Third Owner', 'seats_10', 'seats_2',
     'seats_4', 'seats_5', 'seats_6', 'seats_7', 'seats_8', 'seats_9',
     'seller_type_Dealer', 'seller_type_Trustmark Dealer', 'test1', 'test2',
     'torque', 'transmission_Automatic', 'year']
    num_coulumns = ['year','km_driven',	'mileage','engine','max_power',	'torque','seats','max_torque_rpm']
    for col in columns_models:
        if col not in X_train_cat_ohe.columns:
            if col in num_coulumns :
                X_train_cat_ohe[col] = 0.0
            X_train_cat_ohe[col] = 0.0
    X_train_cat_ohe.columns = X_train_cat_ohe.columns.sort_values()
    X_train_cat_ohe=X_train_cat_ohe.fillna(0.0)
    X_train_cat_ohe=X_train_cat_ohe[columns_models]
    return X_train_cat_ohe

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()]).reset_index()
    prep_df = preprocess_data(df)
    model = load('model.pkl')
    return model.predict(prep_df).round(2)

@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame()
    for item in items:
        df = pd.concat([df, pd.DataFrame([item.dict()])])
    df = preprocess_data(df.reset_index())
    model = load('model.pkl')
    return [round(cast[0], 2) for cast in model.predict(df)]

@app.post("/predict_item_csv")
def upload_data(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file, index_col=0).reset_index()
        prep_df = preprocess_data(df)
        model = load('model.pkl')
        df["forecast"] = model.predict(prep_df).round(2)
        df=df.drop('index', axis=1)
    except Exception:
        raise HTTPException(status_code=500, detail='Something went wrong')
    finally:
        file.file.close()
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return {"message": f"Successfully uploaded {file.filename}"}

#  fastapi dev model.py
# uvicorn.run(app, host="0.0.0.0", port=8000)