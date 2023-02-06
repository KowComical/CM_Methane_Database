import pandas as pd
import os

import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    wetland()


# 湿地
def wetland():
    df_shidi = pd.read_excel(os.path.join(raw_path, '湿地', '湿地_月CH4.xlsx'), sheet_name='月排放').drop(
        columns=['万吨CH4'])
    df_shidi['省'] = df_shidi['省'].str.replace(' ', '')
    # 行转列
    df_shidi = df_shidi.set_index(['省']).stack().reset_index().rename(columns={'level_1': 'month', 0: 'value'})
    df_shidi['value'] = df_shidi['value'] * 10
    df_new_shidi = pd.DataFrame()
    for i in range(start_year, end_year):
        df_shidi['year'] = i
        df_shidi['date'] = pd.to_datetime(df_shidi['year'].astype(str) + '-' + df_shidi['month'].astype(str),
                                          format='%Y-%m月')
        temp = df_shidi[['省', 'date', 'value']]
        df_new_shidi = pd.concat([df_new_shidi, temp]).reset_index(drop=True)
    df_new_shidi = pd.merge(df_new_shidi, df_c, left_on='省', right_on='中文')[['date', '拼音', 'value']].rename(
        columns={'拼音': 'province'})
    df_new_shidi['department'] = 'Wetland'
    df_new_shidi['sector'] = 'Wetland'
    df_new_shidi = df_new_shidi.sort_values('date')
    df_new_shidi.to_csv(os.path.join(out_path, '湿地', '湿地.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
