import pandas as pd
import os

import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    oil_gas()


# 油气
def oil_gas():
    type_list = ['原油', '天然气']
    df_final = pd.DataFrame()
    file_path = os.path.join(raw_path, '油气', '油气1105.xlsx')
    for t in type_list:
        df = pd.read_excel(file_path, sheet_name=t)[:31]
        # 行转列
        df = df.set_index(['平均']).stack().reset_index().rename(
            columns={'level_1': 'date', 0: 'data', '平均': 'province'})
        df['date'] = pd.to_datetime(df['date'], format='%Y年%m月')
        df['data'] = df['data'].replace(' ', 0).astype(float)
        df_pp = df.copy()
        pro_list = df_pp['province'].unique()
        df_new_pp = pd.DataFrame()
        for pro in pro_list:
            temp = df_pp[df_pp['province'] == pro].reset_index(drop=True)
            missing_date = pd.to_datetime(
                pd.date_range(start='%s-01-01' % start_year, end='%s-01-01' % end_year, freq='M').strftime('%Y-%m'))
            needed_date = pd.DataFrame(set(missing_date) - set(temp['date'].tolist()), columns=['date'])
            temp = pd.concat([temp, needed_date]).reset_index(drop=True)
            temp['province'] = pro
            temp['date'] = pd.to_datetime(temp['date'])
            temp = temp.sort_values('date')

            temp['year'] = temp['date'].dt.year
            year_list = temp[temp.isna().any(axis=1)]['year'].unique()
            temp = temp.set_index('date')
            temp['data'] = temp['data'].interpolate(method='linear')
            temp = temp.reset_index()
            if temp['data'].sum() != 0:
                temp_good = temp[~temp['year'].isin(year_list)].reset_index(drop=True)
                for y in year_list:
                    # 拆分年先按照2021年 如果2022年全都是一样的值 则用另外旧的年份
                    year_choose = 2021
                    original_values = temp[temp['year'] == year_choose]['data']
                    while af.all_same(list(original_values)):
                        year_choose -= 1
                        original_values = temp[temp['year'] == year_choose]['data']
                    temp_year = temp[temp['year'] == y].reset_index(drop=True)
                    target_values = temp_year['data']
                    redistributed_values = af.redistribute_monthly_values(original_values, target_values)
                    temp_year['data'] = redistributed_values
                    temp_good = pd.concat([temp_good, temp_year]).reset_index(drop=True)
            else:
                temp_good = temp.copy()
            df_new_pp = pd.concat([df_new_pp, temp_good]).reset_index(drop=True)
            df_new_pp['sector'] = t
        df_final = pd.concat([df_final, df_new_pp]).reset_index(drop=True)

    # 整理并输出
    df_final['department'] = 'Oil and natural gas system'

    df_final['province'] = df_final['province'].str.replace(
        '市', '').str.replace('省', '').str.replace('自治区', '').str.replace(
        '回族', '').str.replace('维吾尔', '').str.replace('壮族', '')
    df_final = pd.merge(df_final, df_c, left_on='province', right_on='中文')[
        ['date', 'data', '拼音', 'sector', 'department']].rename(columns={'拼音': 'province', 'data': 'value'})
    # 统一单位
    df_final['value'] = df_final['value'] / 1000
    # 统一日期
    df_final = df_final[df_final['date'] >= '%s-01-01' % start_year].reset_index(drop=True)
    df_final.to_csv(os.path.join(out_path, '油气', '油气.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
