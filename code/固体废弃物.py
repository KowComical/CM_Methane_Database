import pandas as pd
import os
import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    solid_waste()


def solid_waste():
    col_name = ['年份', '城市', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEPT', 'OCT', 'NOV', 'DEC']
    df = pd.read_excel(os.path.join(raw_path, '固体废弃物', '【monthlyE】中国省级-固体废弃物-万吨-修改DOC.xlsx'))[
        col_name]
    df = df.groupby(['年份', '城市']).sum().reset_index()
    # 数据读取
    df = df.set_index(['年份', '城市']).stack().reset_index().rename(columns={'level_2': 'date', 0: 'value'})
    df['date'] = df['date'].str.title()
    df['date'] = df['date'].replace('Sept', 'Sep')
    try:
        df['年份'] = pd.to_datetime(df['年份'], format='%Y').dt.year
    except:
        df['年份'] = pd.to_datetime(df['年份'], format='%Y-%m-%d').dt.year
    df['date'] = df['年份'].astype(str) + '-' + df['date'].astype(str)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%b')
    df = df[['date', 'value', '城市']]
    df['value'] = df['value'] * 10
    df = df[df['date'] >= '%s-01-01' % start_year].reset_index(drop=True)

    # 改名
    df = df.rename(columns={'城市': 'province'})
    df = pd.merge(df, df_c, left_on='province', right_on='中文')[['date', 'value', '拼音']].rename(
        columns={'拼音': 'province'})
    df_final = af.linear_predict(df, end_year)

    df_final = df_final[['date', 'province', 'value']]
    df_final = df_final.groupby(['date', 'province']).sum().reset_index()
    # 列转行
    df_final['sector'] = '固体废弃物'
    df_final['department'] = 'Solid waste'
    # 输出
    df_final = df_final.sort_values('date')
    df_final.to_csv(os.path.join(out_path, '固体废弃物', '固体废弃物.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
