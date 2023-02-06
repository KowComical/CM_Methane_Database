import pandas as pd
import os
import re
import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    fenbian()
    changdao()


def fenbian():
    # 粪便
    fenbian_path = os.path.join(raw_path, '粪便&肠道', '粪便')
    file_name = af.search_file(fenbian_path)

    year = re.compile(r'- (?P<name>.*?)[.]')
    # 数据清洗
    p = re.compile(r'[\u4e00-\u9fa5]')

    df = pd.DataFrame()
    for f in file_name:
        # 所有sheet_name
        sheet_list = pd.ExcelFile(f).sheet_names
        for s in sheet_list:
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
    df = df[df['date'] >= '%s-01-01' % start_year].reset_index(drop=True)
    df['province'] = df['province'].replace('内蒙', '内蒙古')
    df['value'] = df[['CH4 (kg)', 'CH4 (kg) 驴']].sum(axis=1)
    df = df[['date', 'province', 'type', 'value']]
    # 这里要加个改名
    df = pd.merge(df, df_c, left_on='province', right_on='中文')[['date', 'value', '拼音', 'type']].rename(
        columns={'拼音': 'province'})
    df = df.groupby(['date', 'province']).sum(numeric_only=True).reset_index()
    df['sector'] = 'Manure management'
    df['department'] = 'Livestock'
    df = df.sort_values('date')
    df.to_csv(os.path.join(out_path, '粪便&肠道', '粪便.csv'), index=False, encoding='utf_8_sig')


def changdao():
    # 肠道
    file_path = os.path.join(raw_path, '粪便&肠道', '肠道')

    file = os.path.join(file_path, '肠道发酵 (中国)-20230105.xlsx')

    sheet_list = pd.ExcelFile(file).sheet_names
    df = pd.DataFrame()
    for s in sheet_list:
        r_temp = pd.read_excel(file, sheet_name=s)
        province_list = r_temp.columns[3:-2].tolist()  # 省份名
        # 找到需要数据所在的行
        hang = r_temp[r_temp['Unnamed: 0'] == '甲烷排放量'].index.tolist()[0]
        r_temp = r_temp.iloc[hang:].reset_index(drop=True)

        type_list = r_temp['Unnamed: 1'].dropna().str.replace('牛\n', '').str.replace('羊\n', '')  # 所有动物种类

        # 整理
        r_temp = r_temp[province_list]
        r_temp = r_temp.iloc[:len(type_list)].reset_index(drop=True)
        r_temp['type'] = type_list
        r_temp['year'] = s
        df = pd.concat([df, r_temp]).reset_index(drop=True)
    # 行转列
    df = df.set_index(['type', 'year']).stack().reset_index().rename(columns={'level_2': 'province', 0: 'CH4'})
    year_list = df['year'].unique()
    df_new = pd.DataFrame()
    for y in year_list:
        temp = df[df['year'] == y].reset_index(drop=True)
        temp['year'] = temp['year'].astype(int)
        date_range = pd.date_range(start=y, periods=12, freq='M')
        # 年份范围
        df_date = pd.DataFrame(date_range, columns=['date'])
        df_date['year'] = df_date['date'].dt.year
        df_date['day'] = df_date['date'].dt.day
        # 日在年的占比
        df_sum = df_date.groupby(['year']).sum(numeric_only=True).reset_index().rename(columns={'day': 'sum'})
        df_ratio = pd.merge(df_date, df_sum)
        df_ratio['ratio'] = df_ratio['day'] / df_ratio['sum']
        df_result = pd.merge(temp, df_ratio)
        df_result['CH4'] = df_result['CH4'] * df_result['ratio']
        df_result = df_result[['type', 'date', 'province', 'CH4']]
        df_new = pd.concat([df_new, df_result]).reset_index(drop=True)
    df_new['CH4'] = df_new['CH4'] / 1000 / 1000
    df_new = df_new.drop(columns=['type'])
    df_new = df_new.groupby(['date', 'province']).sum().reset_index()
    df_new['province'] = df_new['province'].str.replace(
        '市', '').str.replace('省', '').str.replace('自治区', '').str.replace(
        '回族', '').str.replace('维吾尔', '').str.replace('壮族', '')
    df_new = pd.merge(df_new, df_c, left_on='province', right_on='中文')[['date', 'CH4', '拼音']].rename(
        columns={'拼音': 'province', 'CH4': 'value'})
    df_new = df_new[df_new['date'] >= '%s-01-01' % start_year].reset_index(drop=True)
    df_new['sector'] = 'Enteric fermentation'
    df_new['department'] = 'Livestock'
    df_new['date'] = pd.to_datetime(pd.to_datetime(df_new['date']).dt.strftime('%Y-%m'))
    df_new = df_new.sort_values('date')
    df_new.to_csv(os.path.join(out_path, '粪便&肠道', '肠道.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
