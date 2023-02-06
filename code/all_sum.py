import pandas as pd
import os
import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()

all_file = af.search_file(out_path)
df_result = pd.DataFrame()
for a in all_file:
    if '部门' not in a:
        temp = pd.read_csv(a)
        if '煤炭' in a:
            temp = temp[temp['date'] >= '%s-01-01' % start_year].reset_index(drop=True).rename(
                columns={'排放因子序号': 'sector', '拼音': 'province'})
            temp = pd.merge(temp, df_c, left_on='province', right_on='中文')[
                ['date', 'value', '拼音', 'sector']].rename(columns={'拼音': 'province'})
            temp['department'] = '煤炭'
        df_result = pd.concat([df_result, temp]).reset_index(drop=True)

# df_result['date'] = pd.to_datetime(pd.to_datetime(df_result['date']).dt.strftime('%Y-%m'))
# df_result = df_result[df_result['date'] >= '%s-01-01' % start_year].reset_index(drop=True)
# df_result.to_csv(os.path.join(out_path, '全部门', '全部门.csv'), index=False, encoding='utf_8_sig')
