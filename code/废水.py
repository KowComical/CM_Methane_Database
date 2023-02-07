import pandas as pd
import os
import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    industry()


def agricultural():
    # 农业废水
    data_path = os.path.join(os.path.join(raw_path, '废水', '农业废水'))
    DcoDa = pd.read_excel(os.path.join(data_path, '农业废水.xlsx'))
    DcoDa = DcoDa.groupby(['year', 'province']).sum().reset_index().rename(columns={'year': 'date'})
    df_all = af.predict_yearly(DcoDa)

    # # ton CH4
    df_all['data'] = df_all['data'] * 0.25 * 0.1  # CH4/COD
    df_all = pd.pivot_table(df_all, index='date', values='data', columns='province').reset_index()
    df_all = df_all.set_index(['date']).stack().reset_index().rename(
        columns={'level_1': 'province', 0: 'agricultural_wastewater'}).rename(
        columns={'date': 'year'})
    df_a = df_all.copy()
    df_a = df_a.groupby(['year', 'province']).mean().reset_index()
    return df_a


def household():
    # 生活废水
    data_path = os.path.join(os.path.join(raw_path, '废水', '生活废水'))
    df_ru = pd.read_excel(os.path.join(data_path, '城镇化率.xlsx'))
    df_ru = af.predict_yearly(df_ru).rename(columns={'data': 'ru'})

    # 读取生活废水
    DCODdom = pd.read_excel(os.path.join(data_path, '生活废水.xlsx')).rename(columns={'year': 'date'})
    DCODdom = af.predict_yearly(DCODdom).rename(columns={'data': 'dom'})
    # # 合并
    df_result = pd.merge(DCODdom, df_ru)
    df_result['DCODdomu'] = df_result['dom'] * df_result['ru'] / 100
    DCODdomu = df_result[['province', 'DCODdomu', 'date']]
    df_result['DCODdomr'] = df_result['dom'] * (1 - (df_result['ru'] / 100))

    # 读取ratio
    ratio = pd.read_excel(os.path.join(data_path, 'ratio.xlsx'))
    o_ratio = ratio.copy()
    # ratio['BOD/COD'] = ratio['BOD/COD']*0.587*0.1

    # 读取污水处理率
    Rdom = pd.read_excel(os.path.join(data_path, '城市污水处理率.xlsx'))
    Rdom = af.predict_yearly(Rdom).rename(columns={'data': 'Rdom'})
    Rdom['Rdom'] = Rdom['Rdom'] / 100
    # 合并
    df_result = pd.merge(df_result, Rdom)
    df_result = pd.merge(df_result, ratio)
    # 直接废水排放
    df_result['EFdomc'] = 0.587 * 0.1
    df_result['Edomd'] = (df_result['DCODdomr'] + df_result['DCODdomu'] * (1 - df_result['Rdom'])) * df_result[
        'EFdomc'] * df_result['BOD/COD']
    df_result.to_csv(os.path.join(data_path, '直接排放废水的甲烷排放估计_t.csv'), index=False, encoding='utf_8_sig')

    # 人口
    POP = pd.read_excel(os.path.join(data_path, '人口统计.xlsx'))
    POP = af.predict_yearly(POP).rename(columns={'data': 'POP'})
    POP['POP'] = POP['POP'] * 10000  # 万人to人
    df_final = pd.merge(df_ru, POP)
    df_final['PCODdomu'] = df_final['ru'] / 100 * df_final['POP'] * 365 * 75 / 1000 / 1000  # g to t

    # 污水
    Rdomc = pd.read_excel(os.path.join(data_path, '污水处理厂集中处理率.xlsx'))
    Rdomc = af.predict_yearly(Rdomc).rename(columns={'data': 'Rdomc'})
    Rdomc['Rdomc'] = Rdomc['Rdomc'] / 100
    # 合并
    df_final = pd.merge(df_final, DCODdomu)
    df_final = pd.merge(df_final, Rdomc)
    df_final = pd.merge(df_final, o_ratio)
    df_final['RCODdomu'] = df_final['PCODdomu'] - df_final['DCODdomu'] * df_final['Rdomc']  # 前后单位差很多 可能有问题
    df_final['Edomc'] = df_final['RCODdomu'] * 0.165 * 0.587 * df_final['BOD/COD']
    df_final = df_final[['date', 'province', 'Edomc']]
    df_final.to_csv(os.path.join(data_path, '生活废水的集中处理方式的甲烷排放估计方法_t.csv'), index=False,
                    encoding='utf_8_sig')

    df_ft = pd.merge(df_ru, POP)
    df_ft['ft'] = df_ft['ru'] * df_ft['POP'] / 100 / 3 / 10000

    df_pt = pd.read_excel(os.path.join(data_path, '城市每万人拥有公厕数.xlsx'))
    df_pt = af.predict_yearly(df_pt).rename(columns={'data': 'pt'})

    EFdomo = pd.merge(df_ft, df_pt)
    EFdomo['EF_1'] = 0.1 * EFdomo['ft'] / (EFdomo['ft'] + EFdomo['pt'])
    EFdomo['EF_2'] = 0.5 * EFdomo['pt'] / (EFdomo['ft'] + EFdomo['pt'])
    EFdomo['EFdomo'] = 0.587 * (EFdomo['EF_1'] + EFdomo['EF_2'])
    Edomdf = pd.merge(Rdom, Rdomc)
    Edomdf = pd.merge(Edomdf, DCODdomu)
    Edomdf = pd.merge(Edomdf, EFdomo)
    Edomdf = pd.merge(Edomdf, o_ratio)

    Edomdf['Edomdf'] = Edomdf['DCODdomu'] * (
            Edomdf['Rdom'] - Edomdf['Rdomc']) * Edomdf['EFdomo'] * Edomdf['BOD/COD'] * Edomdf['ru'] / 100
    Edomdf = Edomdf[['province', 'date', 'Edomdf']]
    Edomdf.to_csv(os.path.join(data_path, '生活废水的其他处理方式的甲烷排放估计方法_t.csv'), index=False,
                  encoding='utf_8_sig')

    df_life_1 = pd.read_csv(os.path.join(data_path, '生活废水的集中处理方式的甲烷排放估计方法_t.csv'))
    df_life_2 = pd.read_csv(os.path.join(data_path, '生活废水的其他处理方式的甲烷排放估计方法_t.csv'))
    df_life = pd.merge(df_life_1, df_life_2)
    df_life['household_wastewater'] = df_life['Edomc'] + df_life['Edomdf']
    df_life = df_life[['date', 'province', 'household_wastewater']].rename(columns={'date': 'year'})
    return df_life


def industry():
    # 工业废水
    data_path = os.path.join(os.path.join(raw_path, '废水', '工业废水'))
    DCOD = pd.read_excel(os.path.join(data_path, 'DCOD.xlsx')).rename(columns={'year': 'date'})
    DCOD = af.predict_yearly(DCOD).rename(columns={'data': 'DCOD'})
    DCOD['DCOD'] = DCOD['DCOD'] * 0.25 * 0.1

    # 工业废水的集中处理部分
    RCOD = pd.read_excel(os.path.join(data_path, 'RCOD.xlsx'))
    RCOD = RCOD.set_index(['province']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'data'})
    RCOD = af.predict_yearly(RCOD).rename(columns={'data': 'RCOD'})
    RCOD['RCOD'] = RCOD['RCOD'] * 0.25 * 0.4674

    df_industry = pd.merge(RCOD, DCOD)
    # df_industry['industrial_wastewater'] = df_industry[['RCOD','DCOD']].sum(axis=1)  # 先不算直接排放
    df_industry['industrial_wastewater'] = df_industry['RCOD']
    df_industry = df_industry[['date', 'province', 'industrial_wastewater']]
    df_year = df_industry.groupby(['date']).sum(numeric_only=True).reset_index().rename(
        columns={'industrial_wastewater': 'sum'})
    df_ratio = pd.merge(df_industry, df_year)
    df_ratio['ratio_industry'] = df_ratio['industrial_wastewater'] / df_ratio['sum']
    df_ratio_industry = df_ratio[['province', 'date', 'ratio_industry']]

    # 月度工业废水甲烷排放估计
    df_type = pd.read_excel(os.path.join(data_path, '各行业工业废水COD排放量.xlsx'))
    df_type = df_type.groupby(['chinese', 'english', 'date']).sum().reset_index()
    type_name = df_type[['chinese', 'english']][~df_type[['chinese', 'english']].duplicated()].reset_index(drop=True)
    df_type = df_type.rename(columns={'english': 'province'}).drop(columns=['chinese'])
    df_type = af.predict_yearly(df_type).rename(columns={'province': 'english'})
    df_type = pd.merge(df_type, type_name)

    df_year = df_type.groupby(['date']).sum(numeric_only=True).reset_index().rename(columns={'data': 'sum'})
    df_ratio = pd.merge(df_type, df_year)
    df_ratio['ratio_type'] = df_ratio['data'] / df_ratio['sum']
    df_ratio_type = df_ratio[['chinese', 'date', 'ratio_type']]
    df_ratio = pd.merge(df_ratio_type, df_ratio_industry)
    df_ratio['ratio'] = df_ratio['ratio_industry'] * df_ratio['ratio_type']
    df_ratio = df_ratio[['date', 'province', 'chinese', 'ratio']]
    df_ratio = pd.pivot_table(df_ratio, index=['date', 'province'], values='ratio', columns='chinese').reset_index()
    df_industry = df_industry.groupby(['date']).sum(numeric_only=True).reset_index()
    df_industry = pd.merge(df_industry, df_ratio)
    # 填充占比
    for d in df_industry.columns[3:]:
        df_industry[d] = df_industry[d] * df_industry['industrial_wastewater']
    df_industry['sum'] = df_industry.iloc[:, 3:].sum(axis=1)
    df_sum = df_industry[['date', 'province', 'sum']]
    df_sum.to_csv(os.path.join(data_path, '工业废水省级年度排放.csv'), index=False, encoding='utf_8_sig')
    # 继续整理
    df_industry = df_industry.drop(columns=['industrial_wastewater', 'sum'])
    df_industry = df_industry.set_index(['date', 'province']).stack().reset_index().rename(
        columns={'level_2': 'type', 0: 'industrial_wastewater'})
    df_data = pd.read_csv(os.path.join(data_path, 'industry_raw.csv')).rename(
        columns={'type': '指标', 'date': '时间', 'data': '值', 'name': '省市名称'})
    # 读取工作日
    work = pd.read_csv(os.path.join(tools_path, 'workday.csv'))
    work['date'] = work['year'].astype(str) + '年' + work['month'].astype(str) + '月'
    # 补全缺失的1&2月当月值
    all_type = df_data['指标'].drop_duplicates().tolist()  # 将所有的类型都提起出来放在列表里
    dangqi_list = [all_type[i] for i, x in enumerate(all_type) if x.find('当') != -1]
    leiji_list = [all_type[i] for i, x in enumerate(all_type) if x.find('累计') != -1]  # 按照当期累计分类成两个列表
    df_result = pd.DataFrame()
    for D, L in zip(dangqi_list, leiji_list):
        df_dangqi = df_data[df_data['指标'] == D]
        df_dangqi = pd.pivot_table(df_dangqi, index='省市名称', values='值', columns='时间').reset_index()
        df_dangqi = df_dangqi.rename(columns={'省市名称': '地区'})

        df_leiji = df_data[df_data['指标'] == L]
        df_leiji = pd.pivot_table(df_leiji, index='省市名称', values='值', columns='时间').reset_index()
        df_leiji = df_leiji.rename(columns={'省市名称': '地区'})

        min_year = int(min(df_dangqi.columns[1:])[:4])
        max_year = int(max(df_dangqi.columns[1:])[:4])
        for i in range(min_year, max_year + 1):  # 按照当前的年份的最小值和最大值来填充1月和2月数据
            df_dangqi['%s年1月' % i] = df_leiji['%s年2月' % i] * work[work['date'] == '%s年1月' % i]['ratio'].values
            df_dangqi['%s年2月' % i] = df_leiji['%s年2月' % i] * work[work['date'] == '%s年2月' % i]['ratio'].values
        df_dangqi['类型'] = D.replace('产量_当月值', '')
        df_dangqi = df_dangqi.set_index(['地区', '类型']).stack().reset_index().rename(
            columns={'level_2': 'date', 0: 'value'})
        df_result = pd.concat([df_result, df_dangqi]).reset_index(drop=True)  # 合并所有结果到新的df
    df_result['类型'] = df_result['类型'].replace('纱', '纱和布').replace('布', '纱和布')
    df_result = df_result.groupby(['地区', '类型', '时间']).sum().reset_index()
    needed_list = ['原煤', '成品糖', '饮料', '纱和布', '机制纸及纸板', '乙烯', '化学纤维']
    df_all = df_result[df_result['类型'].isin(needed_list)].reset_index(drop=True)
    df_all['时间'] = pd.to_datetime(df_all['时间'], format='%Y年%m月')
    df_all['year'] = df_all['时间'].dt.year
    df_sum = df_all.groupby(['year', '地区', '类型']).sum(numeric_only=True).reset_index().rename(
        columns={'value': 'sum'})
    df_ratio = pd.merge(df_sum, df_all)
    df_ratio['ratio'] = df_ratio['value'] / df_ratio['sum']
    df_city = pd.read_csv(os.path.join(tools_path, 'city_name.csv'))

    df_ratio = pd.merge(df_ratio, df_city, left_on='地区', right_on='全称')[['year', '时间', 'ratio', '拼音', '类型']]
    df_ratio['拼音'] = df_ratio['拼音'].replace('InnerMongolia', 'Inner Mongolia')
    df_ratio['类型'] = df_ratio['类型'].replace('原煤', '煤炭开采和洗选业').replace('乙烯',
                                                                                    '化学原料及化学制品制造业').replace(
        '化学纤维', '化学纤维制造业').replace('成品糖', '农副食品加工业')
    df_ratio['类型'] = df_ratio['类型'].replace('机制纸及纸板', '造纸及纸制品业').replace('纱和布', '纺织业').replace(
        '饮料', '酒、饮料和精制茶制造业')
    df_ratio = df_ratio.rename(columns={'拼音': 'province', '类型': 'type', 'year': 'date'})
    p_list = df_ratio['province'].unique()
    t_list = df_ratio['type'].unique()
    d_list = df_ratio['date'].unique()
    df_reversed_ratio = pd.DataFrame()
    for p in p_list:
        for t in t_list:
            for d in d_list:
                temp = df_ratio[
                    (df_ratio['province'] == p) & (df_ratio['date'] == d) & (df_ratio['type'] == t)].reset_index(
                    drop=True)
                if temp['ratio'].sum() == 0:
                    temp['ratio'] = 1 / 12
                df_reversed_ratio = pd.concat([df_reversed_ratio, temp]).reset_index(drop=True)
    df_ratio = df_reversed_ratio.copy()
    # 列转行
    df_new_ratio = pd.DataFrame()
    df_ratio = pd.pivot_table(df_ratio, index=['date', '时间', 'province'], values='ratio',
                              columns='type').reset_index()
    for p in p_list:
        for d in d_list:
            temp = df_ratio[(df_ratio['province'] == p) & (df_ratio['date'] == d)].reset_index(drop=True)
            temp['其他'] = 1 / 12
            df_new_ratio = pd.concat([df_new_ratio, temp]).reset_index(drop=True)
    df_ratio = df_new_ratio.copy()

    df_ratio = df_ratio.set_index(['date', '时间', 'province']).stack().reset_index().rename(
        columns={'level_3': 'type', 0: 'ratio'})

    df_final = pd.merge(df_ratio, df_industry)
    df_final['final'] = df_final['ratio'] * df_final['industrial_wastewater']
    df_final = df_final[['时间', 'province', 'type', 'final']]

    df_final = df_final.groupby(['时间', 'province']).sum(numeric_only=True).reset_index()
    df_final['year'] = df_final['时间'].dt.year
    df_in = df_final.copy()
    df_in = df_in.groupby(['时间', 'province']).mean().reset_index().rename(columns={'final': 'industrial_wastewater'})
    # 合并所有废水
    df_a = agricultural()
    df_life = household()
    df_sum = pd.merge(df_in, df_life)
    df_sum['household_wastewater'] = df_sum['household_wastewater'] / 12
    df_sum = pd.merge(df_sum, df_a)
    df_sum['agricultural_wastewater'] = df_sum['agricultural_wastewater'] / 12
    df_sum['all'] = df_sum['household_wastewater'] + df_sum['industrial_wastewater'] + df_sum['agricultural_wastewater']
    df_sum = df_sum[['时间', 'province', 'household_wastewater', 'agricultural_wastewater', 'industrial_wastewater']].rename(columns={'时间': 'date'})
    # 行转列
    df_sum = df_sum.set_index(['date', 'province']).stack().reset_index().rename(
        columns={'level_2': 'sector', 0: 'value'})
    df_sum['value'] = df_sum['value'] / 1000  # 统一单位
    # 输出
    df_sum['department'] = 'Wastewater'
    df_sum.to_csv(os.path.join(out_path, '废水', '废水.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
