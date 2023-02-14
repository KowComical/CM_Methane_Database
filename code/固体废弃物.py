import pandas as pd
import os
import global_function as af

# 导入全局变量
global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    solid_waste()


def solid_waste():
    col_name = ['年份', '城市', 'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEPT', 'OCT', 'NOV', 'DEC']
    df = pd.read_excel(os.path.join(raw_path, '固体废弃物', '【monthlyE】中国省级-固体废弃物-万吨-修改DOC.xlsx'))
    df = df[col_name]

    # 将宽表转换为长表
    df = df.melt(id_vars=['年份', '城市'], var_name='month', value_name='value')
    df['month'] = df['month'].str.slice(start=0, stop=3)
    df['date'] = pd.to_datetime(df['年份'].astype(str) + df['month'], format='%Y%b')
    df = df[['date', 'value', '城市']].rename(columns={'城市': 'province'})

    # 改名
    df = pd.merge(df, df_c, left_on='province', right_on='中文')[['date', 'value', '拼音']].rename(
        columns={'拼音': 'province'})

    # 数据处理
    df['value'] = df['value'] * 10
    df = df[df['date'] >= f'{start_year}-01-01'].reset_index(drop=True)

    # 数据预测
    df_final = af.linear_predict(df[['date', 'province', 'value']], end_year)

    # 转换为所需输出格式
    df_final = df_final.groupby(['date', 'province'])['value'].sum().reset_index()
    df_final['sector'] = '固体废弃物'
    df_final['department'] = 'Solid waste'
    df_final = df_final.sort_values('date')
    df_final.to_csv(os.path.join(out_path, '固体废弃物', '固体废弃物.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
