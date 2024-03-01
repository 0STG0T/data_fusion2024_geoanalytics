#! /usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime
from pathlib import Path
from typing import Annotated
from warnings import simplefilter

import pandas as pd
import typer
from typer import Option

from catboost import CatBoostClassifier
from statistics import mode 

import h3

import numpy as np

import os


os.environ['_TYPER_STANDARD_TRACEBACK'] = '1'

app = typer.Typer(pretty_exceptions_show_locals=False)

def add_h3_features(df, h3_column_name):
    # Add Area
    df['h3_area'] = df[h3_column_name].apply(lambda x: h3.cell_area(x, unit='m^2'))
    
    # Add Parent and Children Count (specific level can be adjusted)
    parent_level = 8  # One level up
    child_level = 10  # One level down
    df['h3_parent_index'] = df[h3_column_name].apply(lambda x: h3.h3_to_parent(x, parent_level))
    df['parent_lat'] = df['h3_parent_index'].apply(lambda x: h3_to_latlng(x)[0])
    df['parent_lng'] = df['h3_parent_index'].apply(lambda x: h3_to_latlng(x)[1])
    
    df['h3_grandparent_index'] = df['h3_parent_index'].apply(lambda x: h3.h3_to_parent(x, 7))
    df['grandparent_lat'] = df['h3_grandparent_index'].apply(lambda x: h3_to_latlng(x)[0])
    df['grandparent_lng'] = df['h3_grandparent_index'].apply(lambda x: h3_to_latlng(x)[1])
    
    df = df.drop(columns=['h3_grandparent_index', 'h3_parent_index'])

    # Return the modified DataFrame
    return df

# Convert H3 index to latitude and longitude
def h3_to_latlng(h3_index):
  lat, lng = h3.h3_to_geo(h3_index)  # Returns (lat, lng)
  return lat, lng

# Convert latitude and longitude back to H3 index
def latlng_to_h3(lat, lng, resolution=9):
  return h3.geo_to_h3(lat, lng, resolution)

def init_saved_catboost(model_path):
  clf = CatBoostClassifier().load_model(model_path)
  return clf

def preprocess_data(transactions_df: pd.DataFrame):
  # Удаление дубликатов, если они есть. Пример для transactions_df:
  transactions_df = transactions_df.drop_duplicates()
  
  transactions_df['std'] = transactions_df['std'].fillna(0)

  new_trans_df = transactions_df.copy()

  

  transactions_df['trans_lat'] = transactions_df['h3_09'].apply(lambda x: h3_to_latlng(x)[0])
  transactions_df['trans_lng'] = transactions_df['h3_09'].apply(lambda x: h3_to_latlng(x)[1])
  
  transactions_df = add_h3_features(transactions_df, 'h3_09')

  new_trans_df = transactions_df.copy()

  customer_data = new_trans_df.groupby(by='customer_id', as_index=False).agg(
    sum_mean=pd.NamedAgg('sum', 'mean'),
    sum_min=pd.NamedAgg('sum', 'min'),
    sum_max=pd.NamedAgg('sum', 'max'),
    sum_std=pd.NamedAgg('sum', 'std'),
    
    avg_mean=pd.NamedAgg('avg', 'mean'),
    avg_min=pd.NamedAgg('avg', 'min'),
    avg_max=pd.NamedAgg('avg', 'max'),
    avg_std=pd.NamedAgg('avg', 'std'),
    
    min_min = pd.NamedAgg('min', 'min'),
    max_max = pd.NamedAgg('max', 'max'),
    
    count_mean=pd.NamedAgg('count', 'mean'),
    count_min=pd.NamedAgg('count', 'min'),
    count_max=pd.NamedAgg('count', 'max'),
    count_std=pd.NamedAgg('count', 'std'),
    
    cd_mean=pd.NamedAgg('count_distinct', 'mean'),
    cd_min=pd.NamedAgg('count_distinct', 'min'),
    cd_max=pd.NamedAgg('count_distinct', 'max'),
    cd_std=pd.NamedAgg('count_distinct', 'std'),
    
    trans_lat_mean=pd.NamedAgg('trans_lat', 'mean'),
    trans_lat_min=pd.NamedAgg('trans_lat', 'min'),
    trans_lat_max=pd.NamedAgg('trans_lat', 'max'),
    trans_lat_std=pd.NamedAgg('trans_lat', 'std'),
    
    trans_lng_mean=pd.NamedAgg('trans_lng', 'mean'),
    trans_lng_min=pd.NamedAgg('trans_lng', 'min'),
    trans_lng_max=pd.NamedAgg('trans_lng', 'max'),
    trans_lng_std=pd.NamedAgg('trans_lng', 'std'),
    
    parent_lat_mean=pd.NamedAgg('parent_lat', 'mean'),
    parent_lat_min=pd.NamedAgg('parent_lat', 'min'),
    parent_lat_max=pd.NamedAgg('parent_lat', 'max'),
    parent_lat_std=pd.NamedAgg('parent_lat', 'std'),
    
    parent_lng_mean=pd.NamedAgg('parent_lng', 'mean'),
    parent_lng_min=pd.NamedAgg('parent_lng', 'min'),
    parent_lng_max=pd.NamedAgg('parent_lng', 'max'),
    parent_lng_std=pd.NamedAgg('parent_lng', 'std'),
    
    #grandparent_lat_mean=pd.NamedAgg('parent_lat', 'mean'),
    #grandparent_lat_min=pd.NamedAgg('parent_lat', 'min'),
    #grandparent_lat_max=pd.NamedAgg('parent_lat', 'max'),
    #grandparent_lat_std=pd.NamedAgg('parent_lat', 'std'),
    
    #grandparent_lng_mean=pd.NamedAgg('grandparent_lng', 'mean'),
    #grandparent_lng_min=pd.NamedAgg('grandparent_lng', 'min'),
    #grandparent_lng_max=pd.NamedAgg('grandparent_lng', 'max'),
    #grandparent_lng_std=pd.NamedAgg('grandparent_lng', 'std'),
    
    mcc_code_mode=pd.NamedAgg('mcc_code', mode),
    datetime_mode=pd.NamedAgg('datetime_id', mode),
  )

  assert customer_data['customer_id'].nunique() == len(customer_data['customer_id'])
  
  test_data = customer_data.copy()
  
  return test_data

def calculate_probabilities_vectorized(preds_df, hexses_target_lats_lngs, decay_param=0.1):
    # Extract latitudes and longitudes from predictions and hex targets
    atm_lats = preds_df['atm_lat'].values[:, np.newaxis]  # Shape (N, 1) where N is number of ATMs
    atm_lngs = preds_df['atm_lng'].values[:, np.newaxis]  # Shape (N, 1)

    # Convert list of tuples to a 2D NumPy array
    hexses_arr = np.array(hexses_target_lats_lngs)  # Shape (M, 2) where M is number of hexes
    target_lats = hexses_arr[:, 0]  # Shape (M,)
    target_lngs = hexses_arr[:, 1]  # Shape (M,)

    # Compute all distances at once using broadcasting
    # The resulting array will have shape (N, M)
    distances = np.sqrt((atm_lats - target_lats) ** 2 + (atm_lngs - target_lngs) ** 2)

    # Apply decay function to distances to get probabilities
    probas = np.exp(-decay_param * distances)

    # Normalize probabilities so they sum to 1 across each row (for each ATM)
    probas /= probas.sum(axis=1)[:, np.newaxis]  # Normalize across targets for each ATM

    return probas
  
def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    km = 6371 * c 
    return km

def predict_probs(transactions_df, hexses_target):
  test_data = preprocess_data(transactions_df=transactions_df)
  
  meta_catboost_clf = init_saved_catboost('./models/meta_catboost_clf')
  
  def test_data_generator(customer_df, hexses_data, chunk_size=10000):
    unique_atms = pd.DataFrame({'h3_09': hexses_data})
    

    unique_atms['key'] = 1
    customers_chunk_start = 0

    num_chunks = np.ceil(len(customer_df) / chunk_size).astype(int)
    
    for _ in range(num_chunks):

        customer_chunk = customer_df.iloc[customers_chunk_start:customers_chunk_start + chunk_size].copy()
        customer_chunk['key'] = 1  # Temporary key for merging
 
        expanded_chunk = pd.merge(customer_chunk, unique_atms, on='key').drop('key', axis=1)
        
        expanded_chunk['target_lat'] = expanded_chunk['h3_09'].apply(lambda x: h3_to_latlng(x)[0])
        expanded_chunk['target_lng'] = expanded_chunk['h3_09'].apply(lambda x: h3_to_latlng(x)[1])
        
        #expanded_chunk = add_h3_features(expanded_chunk, 'h3_09')
        
        expanded_chunk['mean_distance'] = haversine_vectorized(
            expanded_chunk['trans_lng_mean'],
            expanded_chunk['trans_lat_mean'],
            expanded_chunk['target_lng'],
            expanded_chunk['target_lat']
        )
        
        expanded_chunk['min_distance'] = haversine_vectorized(
            expanded_chunk['trans_lng_min'],
            expanded_chunk['trans_lat_min'],
            expanded_chunk['target_lng'],
            expanded_chunk['target_lat']
        )
        
        expanded_chunk['max_distance'] = haversine_vectorized(
            expanded_chunk['trans_lng_max'],
            expanded_chunk['trans_lat_max'],
            expanded_chunk['target_lng'],
            expanded_chunk['target_lat']
        )
        
  
        customers_chunk_start += chunk_size
        
        yield expanded_chunk
        
  test_generator = test_data_generator(customer_df=test_data, hexses_data=hexses_target, chunk_size=1000)
  
  flag = 0

  preds_df = pd.DataFrame({})

  for chunk in test_generator:
      preds = meta_catboost_clf.predict_proba(chunk.drop(columns=['customer_id', 'h3_09']))
      preds_df_chunk = pd.DataFrame({'customer_id': chunk['customer_id'], 'h3_09': chunk['h3_09'], 'proba': [x[1] for x in preds]})
      
      if not flag:
          preds_df = preds_df_chunk
          flag += 1
      else:
          preds_df = pd.concat((preds_df.reindex(), preds_df_chunk.reindex()), axis=0)
          
          flag += 1
          
      print(flag)
      
  preds_df['proba'] = preds_df.groupby('customer_id')['proba'].transform(lambda x: x / x.sum())
  
  submit_df = preds_df.pivot(index='customer_id', columns='h3_09', values='proba').reset_index()

  missing_cols = set(hexses_target) - set(submit_df.columns)
  for col in missing_cols:
      submit_df[col] = 0.0  

  submit_df = submit_df[['customer_id'] + sorted(set(hexses_target))]
  
  print(submit_df)
  
  submit_df = submit_df.rename_axis(None, axis=1)
  
  return submit_df
  
def main(
  hexses_target_path: Annotated[
    Path, Option('--hexses-target-path', '-ht', dir_okay=False, help='Список локаций таргета', show_default=True, exists=True)
  ] = 'hexses_target.lst',
  hexses_data_path: Annotated[
    Path, Option('--hexses-data-path', '-hd', dir_okay=False, help='Список локаций транзакций', show_default=True, exists=True)
  ] = 'hexses_data.lst',
  input_path: Annotated[
    Path, Option('--input-path', '-i', dir_okay=False, help='Входные данные', show_default=True, exists=True)
  ] = 'moscow_transaction_data01.parquet',
  output_path: Annotated[
    Path, Option('--output-path', '-o', dir_okay=False, help='Выходные данные', show_default=True)
  ] = 'output.parquet',
):
    with open(hexses_target_path, "r") as f:
        hexses_target = [x.strip() for x in f.readlines()]
    with open(hexses_data_path, "r") as f:
        hexses_data = [x.strip() for x in f.readlines()]
        
    transactions_df = pd.read_parquet(input_path)
    
    customers = pd.read_parquet(input_path).customer_id.unique()
    
    submit_df = predict_probs(transactions_df=transactions_df, hexses_target=hexses_target)
    
    submit_df.to_parquet(output_path)
    
    

if __name__ == '__main__':
  typer.run(main, )
