import pandas as pd
import os

import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    wetland()


# 湿地
def wetland():
    df_shidi = pd.read_excel(os.path.join(raw_path, '湿地', '湿地月排放-千吨.xlsx'), header=1)
    # 整理数据
    df_shidi['date'] = pd.to_datetime(df_shidi['年月'], format='%Y-%m月')
    df_shidi = df_shidi.drop(columns=['年月', '年', '月'])

    # 行转列
    df_shidi = df_shidi.set_index(['date']).stack().reset_index().rename(columns={'level_1': 'province', 0: 'value'})
    # 整理省份
    df_shidi['province'] = df_shidi['province'].str.replace(' ', '')
    df_shidi = pd.merge(df_shidi, df_c, left_on='province', right_on='中文')[['date', '拼音', 'value']].rename(
        columns={'拼音': 'province'})
    df_shidi['department'] = 'Wetland'
    df_shidi['sector'] = 'Wetland'
    df_shidi = df_shidi.sort_values('date')
    df_shidi.to_csv(os.path.join(out_path, '湿地', '湿地.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
