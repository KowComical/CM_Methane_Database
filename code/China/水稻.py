import pandas as pd
import os
import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    rice_cultivation()


# 水稻
def rice_cultivation():
    file_path = os.path.join(raw_path, '水稻', '中国水稻甲烷排放计算_2022.xlsx')

    sheet_list = pd.ExcelFile(file_path).sheet_names
    sheet_list = [sheet_list[i] for i, x in enumerate(sheet_list) if x.find('20') != -1]

    df = pd.DataFrame()
    for s in sheet_list:
        temp = pd.read_excel(file_path, sheet_name=s, header=1).iloc[:, :8]
        temp['year'] = s[:4]
        # 整理
        temp = temp.set_index(['地区', 'year']).stack().reset_index().rename(columns={'level_2': 'month', 0: 'coal'})
        temp['date'] = pd.to_datetime(temp['year'] + '-' + temp['month'], format='%Y-%m月')
        # 填充缺失的月份
        temp = pd.pivot_table(temp, index='date', values='coal', columns='地区').reset_index()
        date_range = pd.to_datetime(pd.date_range(start=s[:4], periods=12, freq='M').strftime('%Y-%m'))
        needed_date = pd.DataFrame(set(date_range) - set(temp['date'].tolist()), columns=['date'])
        # 合并
        temp = pd.concat([temp, needed_date]).reset_index(drop=True).fillna(0)
        df = pd.concat([df, temp]).reset_index(drop=True)
    # 行转列
    df = df.set_index(['date']).stack().reset_index().rename(columns={'level_1': 'province', 0: 'value'})
    df['value'] = df['value'] / 1000
    df = df[df['date'] >= '%s-01-01' % start_year].reset_index(drop=True)
    # 统一列名
    df['department'] = 'Rice cultivation'
    df['sector'] = 'Rice cultivation'
    # 统一省份名
    df['province'] = df['province'].str.replace('市', '').str.replace('省', '').str.replace('自治区', '').str.replace(
        '回族', '').str.replace('维吾尔', '').str.replace('壮族', '')
    df = pd.merge(df, df_c, left_on='province', right_on='中文')[
        ['date', 'value', '拼音', 'sector', 'department']].rename(columns={'拼音': 'province'})

    # 输出
    df = df.sort_values('date')
    df.to_csv(os.path.join(out_path, '水稻', '水稻.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
