import pandas as pd
import os
from datetime import datetime

# 路径
global_path = 'K:\\Github\\CM_Methane_Database\\data\\China\\'
raw_path = os.path.join(global_path, 'raw')
tools_path = os.path.join(global_path, 'tools')
out_path = os.path.join(global_path, 'cleaned')
# 参数
df_c = pd.read_csv(os.path.join(tools_path, 'city_name.csv'))
end_year = int(datetime.now().strftime('%Y'))
start_year = 2013


def main():
    energy()


def energy():
    file = os.path.join(raw_path, '能源燃烧', 'energy combustion.xlsx')
    sheet_list = pd.ExcelFile(file).sheet_names

    df_result = pd.DataFrame()
    for s in sheet_list:
        if s == 'SUM':
            temp = pd.read_excel(file, sheet_name=s)
            temp = temp.set_index(['万吨CH4']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'value'})
            temp['value'] = temp['value'] * 10
            temp['date'] = pd.to_datetime(temp['date'], format='%Y_%m')
            temp['万吨CH4'] = temp['万吨CH4'].replace('内蒙', '内蒙古')
            temp = pd.merge(temp, df_c, left_on='万吨CH4', right_on='中文')[['date', 'value', '拼音']].rename(
                columns={'拼音': 'province'})
            df_result = pd.concat([df_result, temp]).reset_index(drop=True)
        else:
            temp = pd.read_excel(file, sheet_name=s)
            temp = temp.set_index(['年月']).stack().reset_index().rename(
                columns={'level_1': 'province', 0: 'value'}).rename(columns={'年月': 'date'})
            temp['value'] = temp['value'] * 10
            temp['date'] = pd.to_datetime(temp['date'], format='%Y%m')
            df_result = pd.concat([df_result, temp]).reset_index(drop=True)
    df_result = df_result.groupby(['date', 'province']).sum().reset_index()
    df_result = df_result[df_result['date'] >= '%s-01-01' % start_year].reset_index(drop=True)
    df_result['sector'] = '能源燃烧'
    df_result['department'] = 'Energy combustion'
    df_result = df_result.sort_values('date')
    df_result.to_csv(os.path.join(out_path, '能源燃烧', '能源燃烧.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
