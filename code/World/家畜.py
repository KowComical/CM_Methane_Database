import re
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
import numpy as np


# 粪便
def search_file(path):
    import os
    import sys
    sys.dont_write_bytecode = True
    file = []
    for parent, surnames, filenames in os.walk(path):
        for fn in filenames:
            file.append(os.path.join(parent, fn))
    return file


file_path = 'K:\\Github\\CM_Methane_Database\\data\\World\\raw\\家畜\\粪便\\'
out_path = 'K:\\Github\\CM_Methane_Database\\data\\World\\cleaned\\家畜\\'
# 所有文件
file_name = search_file(file_path)

year = re.compile(r'- (?P<name>.*?)[.]')
# 数据清洗
p = re.compile(r'[\u4e00-\u9fa5]')

df = pd.DataFrame()
for f in file_name:
    # 所有sheet_name
    sheet_list = pd.ExcelFile(f).sheet_names
    for s in sheet_list:
        if s != 'Sheet1':
            temp = pd.read_excel(f, sheet_name=s)
            # 找到各动物的行
            first_hang = temp[temp['Unnamed: 0'] == 'CH4 Density'].index.tolist()[0]
            new_data = []
            mydata = temp.iloc[first_hang:]['Unnamed: 0'].dropna().tolist()
            for i in range(len(mydata)):
                data = re.findall(p, mydata[i])
                result = ''.join(data)
                new_data.append(result)
            type_list = []
            for i in range(len(new_data)):
                if new_data[i] != '':
                    type_list.append(new_data[i])
            for t in type_list:
                if t == '驴骡':
                    t = '驴/骡'
                header_num = temp[temp['Unnamed: 0'] == t].index.values[-1] + 1
                # 找到需要的列名
                real_temp = pd.read_excel(f, header=header_num, sheet_name=s)
                col_list = real_temp.loc[:, real_temp.columns.str.contains('CH4', case=False)].columns.tolist()
                col_list = [col_list[i] for i, x in enumerate(col_list) if not x.find('potential') != -1]  # 清除不需要的列名
                if t == '驴/骡':
                    for c in col_list:
                        temp_col = ['月份.1'] + [c]
                        part_temp = real_temp[temp_col][:12]
                        part_temp['year'] = year.findall(f)[0]
                        part_temp['type'] = re.findall(r'([^ ]*)$', part_temp.columns[1])[0]
                        part_temp['province'] = s
                        part_temp = part_temp.rename(columns={part_temp.columns[1]: 'CH4 (kg)'})
                        df = pd.concat([df, part_temp]).reset_index(drop=True)
                else:
                    needed_col = ['月份.1'] + col_list
                    needed_temp = real_temp[needed_col][:12]  # 只要前12行
                    # 添加国家和动物种类
                    needed_temp['year'] = year.findall(f)[0]
                    needed_temp['type'] = t
                    needed_temp['province'] = s
                    df = pd.concat([df, needed_temp]).reset_index(drop=True)

df['月份.1'] = df['月份.1'].astype(int)
df['date'] = pd.to_datetime(df['year'] + df['月份.1'].astype(str), format='%Y%m')
df = df.drop(columns=['月份.1', 'year'])
df['CH4 (kg)'] = df['CH4 (kg)'] / 1000 / 1000
# 列转行
df = pd.pivot_table(df, index=['date', 'province'], values='CH4 (kg)', columns='type').reset_index()
# 最后整理
df['value'] = df.sum(axis=1, numeric_only=True)
df = df[['date', 'province', 'value']]
df = df.groupby(['date', 'province']).sum().reset_index()
df['sector'] = 'Manure management'
df['department'] = 'Livestock'
