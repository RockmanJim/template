import pandas as pd
import lightgbm as lgb
import os
from typing import Dict, Iterable, Tuple, List
import numpy as np
from multiprocessing import Pool
from functools import reduce
import time


DATA_BASE = './components/data'
FILE_BASE = './components/'

PRE_INFO = os.path.join(FILE_BASE, 'fl.csv')
FILL_DATA = os.path.join(FILE_BASE, 'fill_data.txt')
MODEL_FILE = os.path.join(FILE_BASE, 'my_model.txt')

def mode(s: pd.Series) -> float:
    return s.mode()[0]

FUNC_DICT = {
    'max': 'max',
    'min': 'min',
    'mean': 'mean',
    'median': np.median,
    'mode': mode
}


def get_fill_data(fill_data: str) -> Dict[str, float]:
    with open(fill_data, 'r', encoding='utf-8') as f:
        d = eval(f.read())
    return d


def get_pre_info(pre_info: str) -> pd.DataFrame:
    df = pd.read_csv(pre_info)
    df['files'] = df['path'].transform(
        lambda x: os.listdir(os.path.join(DATA_BASE, x)))
    return df


def get_recode_single(path: str, calculations: List[str], full_feature_name: str, file: str) -> pd.DataFrame:
    df = pd.read_csv(os.path.join(DATA_BASE, path, file))
    if len(df):
        df_ana = df.agg([FUNC_DICT[c] for c in calculations])
    else:
        df_ana = pd.DataFrame({
            df.columns[0]: [np.nan] * len(calculations)
        }, index=calculations)

    df_name = file[:file.find('.csv')]
    df_ana.columns = [df_name, ]

    return df_ana.T.rename(columns={
        'max': f'{full_feature_name}__max',
        'min': f'{full_feature_name}__min',
        'mean': f'{full_feature_name}__mean',
        'median': f'{full_feature_name}__median',
        'mode': f'{full_feature_name}__mode',
    })


def get_feature_single(args: Tuple[int, pd.Series]) -> pd.DataFrame:
    _, s = args
    print(s['feature'])
    df: pd.DataFrame = pd.concat(
        get_recode_single(s['path'], eval(s['calculations']),
                          f'{s["P"]}__{s["feature"]}', file)
        for file in s['files']
    )
    return df.reset_index().rename(columns={'index': 'ID'})


def get_feture_all(iter: Iterable[Tuple[int, pd.Series]]) -> pd.DataFrame:
    with Pool() as p:
        print('merging data...')
        outputs: pd.DataFrame = reduce(
            lambda left_df, right_df: pd.merge(
                left_df,
                right_df,
                'outer',
                'ID'
            ),
            p.map(
                get_feature_single,
                iter
            )
        )
    print(f'data shape: {outputs.shape}')
    return outputs


if __name__ == '__main__':
    t = time.time()
    fill_data = get_fill_data(FILL_DATA)

    info = get_pre_info(PRE_INFO)

    df = get_feture_all(info.iterrows())

    df.fillna(fill_data, inplace=True)

    model = lgb.Booster(model_file=MODEL_FILE)

    result = pd.DataFrame({
        'id': df['ID'],
        'result': model.predict(df.drop(columns='ID'))
    })
    result['result'] = result['result'].transform(lambda x: round(x))

    result.to_csv('test_predict.csv', index=False)
    print(f'Time: {time.time()-t}')