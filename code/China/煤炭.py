import pandas as pd
import os
import numpy as np
from sklearn.linear_model import LinearRegression

import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    coal()


def process(df_efp, df_pp, num, df_province):
    # 统一省份名
    df_province = df_province.rename(columns={'缩写': 'Province'})
    df_efp = pd.merge(df_efp, df_province)[['中文', 2011]]
    df_pp = pd.merge(df_province, df_pp, left_on='全称', right_on='name', how='right')[['中文', 'date', 'carbon']]

    df_ratio = pd.read_excel(os.path.join(raw_path, '煤炭', '2010年中国矿井和露天矿比例.xlsx'), header=1)
    df_ratio = df_ratio.rename(columns={'Unnamed: 0': 'province'})
    df_ratio = df_ratio[['province', '国有重点煤矿原煤露天产量(吨)', '国有重点煤矿原煤矿井产量(吨)']].replace('--', 0)
    df_ratio['ratio_前'] = df_ratio['国有重点煤矿原煤矿井产量(吨)'] / (
            df_ratio['国有重点煤矿原煤露天产量(吨)'] + df_ratio['国有重点煤矿原煤矿井产量(吨)'])
    df_ratio['ratio_前'] = df_ratio['ratio_前'].fillna(1)

    df_ratio['ratio_后'] = df_ratio['国有重点煤矿原煤露天产量(吨)'] / (
            df_ratio['国有重点煤矿原煤露天产量(吨)'] + df_ratio['国有重点煤矿原煤矿井产量(吨)'])
    df_ratio['ratio_后'] = df_ratio['ratio_后'].fillna(1)

    df_ratio['province'] = df_ratio['province'].str.replace('市', '').str.replace('省', '')
    df_ratio = df_ratio.rename(columns={'province': '中文'})
    # 计算
    df_result = pd.merge(df_pp, df_efp, how='left')
    df_result = pd.merge(df_result, df_ratio, how='left').fillna(0)
    # 统一单位
    df_result['carbon'] = df_result['carbon'] * 10000  # 万吨变为吨

    # 前
    df_result['地下矿开采甲烷排放'] = df_result['carbon'] * df_result[2011] * df_result['ratio_前'] * 0.67
    df_result['地下矿采后甲烷排放'] = df_result['carbon'] * df_result['ratio_前'] * 0.67 * (0.73 * 0.9 + 0.27 * 3)
    # 后
    df_result['露天矿开采后甲烷排放_EFp_s-1'] = df_result['carbon'] * df_result['ratio_后'] * 0.67 * 0.1
    df_result['露天矿开采后甲烷排放_EFp_s-2'] = df_result['carbon'] * df_result['ratio_后'] * 0.67 * 0.5

    df_result['露天矿开采甲烷排放_EFp_s-1'] = df_result['carbon'] * df_result['ratio_后'] * 0.67 * 1.2
    df_result['露天矿开采甲烷排放_EFp_s-2'] = df_result['carbon'] * df_result['ratio_后'] * 0.67 * 2
    #
    df_result['date'] = df_result['date'].astype(str)
    df_result_1 = df_result[['中文', 'date', '地下矿开采甲烷排放']]
    df_result_1 = pd.pivot_table(df_result_1, index='中文', values='地下矿开采甲烷排放', columns='date').reset_index()
    df_result_1.to_csv(os.path.join(raw_path, '煤炭', '地下矿开采甲烷排放.csv'), index=False, encoding='utf_8_sig')

    df_result_2 = df_result[['中文', 'date', '地下矿采后甲烷排放']]
    df_result_2 = pd.pivot_table(df_result_2, index='中文', values='地下矿采后甲烷排放', columns='date').reset_index()
    df_result_2.to_csv(os.path.join(raw_path, '煤炭', '地下矿采后甲烷排放.csv'), index=False, encoding='utf_8_sig')

    # 后
    df_result_3 = df_result[['中文', 'date', '露天矿开采后甲烷排放_EFp_s-1']]
    df_result_4 = df_result[['中文', 'date', '露天矿开采后甲烷排放_EFp_s-2']]

    df_result_3 = pd.pivot_table(df_result_3, index='中文', values='露天矿开采后甲烷排放_EFp_s-1', columns='date')
    df_result_3.reset_index().to_csv(os.path.join(raw_path, '煤炭', '露天矿开采后甲烷排放_EFp_s-1.csv'), index=False,
                                     encoding='utf_8_sig')

    df_result_4 = pd.pivot_table(df_result_4, index='中文', values='露天矿开采后甲烷排放_EFp_s-2', columns='date')
    df_result_4.reset_index().to_csv(os.path.join(raw_path, '煤炭', '露天矿开采后甲烷排放_EFp_s-2.csv'), index=False,
                                     encoding='utf_8_sig')

    df_result_5 = df_result[['中文', 'date', '露天矿开采甲烷排放_EFp_s-1']]
    df_result_6 = df_result[['中文', 'date', '露天矿开采甲烷排放_EFp_s-2']]

    df_result_5 = pd.pivot_table(df_result_5, index='中文', values='露天矿开采甲烷排放_EFp_s-1', columns='date')
    df_result_5.reset_index().to_csv(os.path.join(raw_path, '煤炭', '露天矿开采甲烷排放_EFp_s-1.csv'), index=False,
                                     encoding='utf_8_sig')

    df_result_6 = pd.pivot_table(df_result_6, index='中文', values='露天矿开采甲烷排放_EFp_s-2', columns='date')
    df_result_6.reset_index().to_csv(os.path.join(raw_path, '煤炭', '露天矿开采甲烷排放_EFp_s-2.csv'), index=False,
                                     encoding='utf_8_sig')

    ep = pd.read_csv(os.path.join(raw_path, '煤炭', '地下矿开采甲烷排放.csv'))

    ep_s_1 = pd.read_csv(os.path.join(raw_path, '煤炭', '露天矿开采甲烷排放_EFp_s-1.csv'))
    ep_s_2 = pd.read_csv(os.path.join(raw_path, '煤炭', '露天矿开采甲烷排放_EFp_s-2.csv'))
    ep_pu = pd.read_csv(os.path.join(raw_path, '煤炭', '地下矿采后甲烷排放.csv'))
    ep_ps_1 = pd.read_csv(os.path.join(raw_path, '煤炭', '露天矿开采后甲烷排放_EFp_s-1.csv'))
    ep_ps_2 = pd.read_csv(os.path.join(raw_path, '煤炭', '露天矿开采后甲烷排放_EFp_s-2.csv'))

    ep = ep.set_index(['中文']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'ep'})
    ep_s_1 = ep_s_1.set_index(['中文']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'ep_s_1'})
    ep_s_2 = ep_s_2.set_index(['中文']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'ep_s_2'})
    ep_pu = ep_pu.set_index(['中文']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'ep_pu'})
    ep_ps_1 = ep_ps_1.set_index(['中文']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'ep_ps_1'})
    ep_ps_2 = ep_ps_2.set_index(['中文']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'ep_ps_2'})

    eab = pd.merge(ep, ep_s_1)
    eab = pd.merge(eab, ep_s_2)
    eab = pd.merge(eab, ep_pu)
    eab = pd.merge(eab, ep_ps_1)
    eab = pd.merge(eab, ep_ps_2)

    eab['1_1'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_1']].sum(axis=1) / 99
    eab['1_2'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_2']].sum(axis=1) / 99
    eab['2_1'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_1']].sum(axis=1) / 99
    eab['2_2'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_2']].sum(axis=1) / 99

    eab.to_csv(os.path.join(raw_path, '煤炭', '废弃煤矿.csv'), index=False, encoding='utf_8_sig')

    # 16组数据
    ept = pd.DataFrame()

    ept['1_1_1_1'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_1', '1_1']].sum(axis=1)
    ept['1_1_1_2'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_1', '1_2']].sum(axis=1)
    ept['1_1_2_1'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_1', '2_1']].sum(axis=1)
    ept['1_1_2_2'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_1', '2_2']].sum(axis=1)
    ept['1_2_1_1'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_2', '1_1']].sum(axis=1)
    ept['1_2_1_2'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_2', '1_2']].sum(axis=1)
    ept['1_2_2_1'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_2', '2_1']].sum(axis=1)
    ept['1_2_2_2'] = eab[['ep', 'ep_s_1', 'ep_pu', 'ep_ps_2', '2_2']].sum(axis=1)
    ept['2_1_1_1'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_1', '1_1']].sum(axis=1)
    ept['2_1_1_2'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_1', '1_2']].sum(axis=1)
    ept['2_1_2_1'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_1', '2_1']].sum(axis=1)
    ept['2_1_2_2'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_1', '2_2']].sum(axis=1)
    ept['2_2_1_1'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_2', '1_1']].sum(axis=1)
    ept['2_2_1_2'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_2', '1_2']].sum(axis=1)
    ept['2_2_2_1'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_2', '2_1']].sum(axis=1)
    ept['2_2_2_2'] = eab[['ep', 'ep_s_2', 'ep_pu', 'ep_ps_2', '2_2']].sum(axis=1)
    ept['avg'] = ept.mean(axis=1)
    ept['province'] = eab['中文']
    ept['date'] = eab['date']

    ept.to_csv(os.path.join(raw_path, '煤炭', '月度总排放(Et).csv'), index=False, encoding='utf_8_sig')
    ept['date'] = pd.to_datetime(ept['date'])
    ept['year'] = ept['date'].dt.year

    # 方法2
    # 先将省级的月度总排放加总求和，得到全国的年度煤炭部门的甲烷总排放
    et = ept.groupby(['year']).sum(numeric_only=True).reset_index()[['year', 'avg']]
    et.to_csv(os.path.join(raw_path, '煤炭', '全国每年2020-2021甲烷总排放_方法二.csv'), index=False,
              encoding='utf_8_sig')

    # 求全国年度煤炭部门甲烷净排放
    ratio = pd.read_excel(os.path.join(raw_path, '煤炭', '国家尺度煤层气瓦斯利用量.xlsx'),
                          header=2).rename(
        columns={'会议年份': 'year'}).drop(columns=['井下瓦斯利用量', '地面煤层气利用量'])

    s_year = max(ratio['year']) + 1
    df_predicted = pd.DataFrame()

    for i in range(s_year, 2023):
        X = ratio['year'].values.reshape(-1, 1)  # put your dates in here
        y = ratio['煤层气瓦斯利用量'].values.reshape(-1, 1)  # put your kwh in here

        model = LinearRegression()
        model.fit(X, y)

        X_predict = pd.DataFrame([i]).values.reshape(-1, 1)  # put the dates of which you want to predict kwh here
        y_predict = model.predict(X_predict)
        # 将结果加入df中
        predict = pd.DataFrame([[int(X_predict), float(y_predict)]], columns=ratio.columns)
        ratio = pd.concat([ratio, predict]).reset_index(drop=True)
        ratio = pd.concat([df_predicted, ratio]).reset_index(drop=True)

    ent = pd.merge(et, ratio)
    ent['ent'] = ent['avg'] - (ent['煤层气瓦斯利用量'] * 100000000 * 0.67)
    # ent['ent'] = ent['avg']  # 那个之前你帮我弄的煤炭甲烷排放，其中有减去一分布回收（利用）量 那部分你能帮我去掉重新算一下吗？
    ent = ent[['year', 'ent']]

    ent.to_csv(os.path.join(raw_path, '煤炭', '全国从2010-2021年每年的净甲烷排放_方法二.csv'), index=False,
               encoding='utf_8_sig')

    # 对Ent做省级分配，按照各省年度的Eapt占Et的比例，对Ent进行省级分配，分配到Enpt
    enpt = ept[['province', 'avg', 'year']].groupby(['year', 'province']).sum().reset_index().rename(
        columns={'avg': 'enpt'})
    enpt = pd.merge(enpt, et)
    enpt['ratio'] = enpt['enpt'] / enpt['avg']
    enpt = enpt[['year', 'province', 'ratio']]
    enpt = pd.merge(enpt, ent)
    enpt['enpt'] = enpt['ratio'] * enpt['ent']

    enpt.to_csv(os.path.join(raw_path, '煤炭', '各省从2010-2021年每年的净甲烷排放_方法二.csv'), index=False,
                encoding='utf_8_sig')

    # 对Enpt进行月度分配，按之前各个省的月度排放与各省年排放的比例 Ept/Eapt进行月度分配，得到各省的月度的Enmpt
    ept['month'] = ept['date'].dt.month
    enmpt = ept[['province', 'avg', 'date', 'year', 'month']].groupby(
        ['date', 'province', 'year', 'month']).sum().reset_index().rename(columns={'avg': 'enmpt'})
    enmpt = pd.merge(enmpt, et)
    enmpt['ratio'] = enmpt['enmpt'] / enmpt['avg']
    enmpt = enmpt[['year', 'month', 'province', 'ratio']]
    enmpt = pd.merge(enmpt, ent)
    enmpt['enmpt'] = enmpt['ratio'] * enmpt['ent']
    enmpt = enmpt.drop(columns=['ratio', 'ent'])

    enmpt.to_csv(os.path.join(raw_path, '煤炭', '各省从2010年1月-2021年12月每月的净甲烷排放_方法二.csv'), index=False,
                 encoding='utf_8_sig')

    enmpt['date'] = pd.to_datetime(enmpt[['year', 'month']].assign(Day=1))  # 合并年月

    # 列转行
    enmpt = pd.pivot_table(enmpt, index='date', values='enmpt', columns='province').reset_index()

    enmpt.to_csv(os.path.join(raw_path, '煤炭', 'result_%s.csv' % num), index=False, encoding='utf_8_sig')


# 煤炭
# 先下载源数据
def coal():
    file_path = os.path.join(raw_path, '煤炭')
    file_name = af.search_file(file_path)
    dangqi_path = [file_name[i] for i, x in enumerate(file_name) if x.find('当期') != -1][0]
    leiji_path = [file_name[i] for i, x in enumerate(file_name) if x.find('累计') != -1][0]

    # df_test = pd.read_csv(dangqi_path)
    # code_list = ['A03010101','A03010102']
    # name_list = ['原煤产量当期值(万吨)','原煤产量累计值(万吨)']
    # for k,n in zip(code_list,name_list):
    #     df_result = get_json('fsyd',k,'2000-%s' % end_year)
    #     df_result['能源'] = n
    #     df_result.to_csv(os.path.join(file_path,'%s.csv' % n), index=False, encoding='utf_8_sig')

    # 将源数据整理一下
    df_dangqi = pd.read_csv(dangqi_path)
    df_leiji = pd.read_csv(leiji_path)

    df_dangqi = pd.pivot_table(df_dangqi, index='name', values='data', columns='date').reset_index()
    df_leiji = pd.pivot_table(df_leiji, index='name', values='data', columns='date').reset_index()

    # 读取工作日
    work = pd.read_csv(os.path.join(tools_path, 'workday.csv'))
    work['date'] = work['year'].astype(str) + '年' + work['month'].astype(str) + '月'

    # 补全缺失的1&2月当月值
    min_year = int(min(df_dangqi.columns[1:])[:4])
    max_year = int(max(df_dangqi.columns[1:])[:4])
    for i in range(min_year, max_year + 1):  # 按照当前的年份的最小值和最大值来填充1月和2月数据
        df_dangqi['%s年1月' % i] = df_leiji['%s年2月' % i] * work[work['date'] == '%s年1月' % i]['ratio'].values
        df_dangqi['%s年2月' % i] = df_leiji['%s年2月' % i] * work[work['date'] == '%s年2月' % i]['ratio'].values

    df_pp = df_dangqi.set_index(['name']).stack().reset_index().rename(columns={'level_1': 'date', 0: 'carbon'})
    df_pp['date'] = pd.to_datetime(df_pp['date'], format='%Y年%m月')
    df_pp = df_pp.sort_values('date').reset_index(drop=True)
    pro_list = df_pp['name'].unique()
    df_new_pp = pd.DataFrame()
    for pro in pro_list:
        temp = df_pp[df_pp['name'] == pro].reset_index(drop=True)
        if '西藏自治区' != pro:
            if temp['carbon'].sum() != 0:
                temp = temp.replace(0, np.nan)
                # 将含有null值行的年份找到
                temp['year'] = temp['date'].dt.year
                year_list = temp[temp.isna().any(axis=1)]['year'].unique()
                # 填充缺失值
                temp["carbon"].interpolate(method='linear', inplace=True)
                # 将含有null值的年份再按照前一年重新分配一下
                temp_good = temp[~temp['year'].isin(year_list)].reset_index(drop=True)
                for y in year_list:
                    # 拆分年先按照2022年 如果2022年全都是一样的值 则用另外旧的年份
                    year_choose = 2022
                    original_values = temp[temp['year'] == year_choose]['carbon']
                    while af.all_same(list(original_values)):
                        year_choose -= 1
                        original_values = temp[temp['year'] == year_choose]['carbon']
                    temp_year = temp[temp['year'] == y].reset_index(drop=True)
                    target_values = temp_year['carbon']
                    redistributed_values = af.redistribute_monthly_values(original_values, target_values)
                    temp_year['carbon'] = redistributed_values
                    temp_good = pd.concat([temp_good, temp_year]).reset_index(drop=True)
            else:
                temp_good = temp.copy()
        else:
            temp_good = temp.copy()
        df_new_pp = pd.concat([df_new_pp, temp_good]).reset_index(drop=True)

    df_pp = df_new_pp.copy()
    df_pp = df_pp.drop(columns=['year'])

    # 读取排放因子
    ef_path = os.path.join(raw_path, '煤炭', '中国井下煤炭甲烷排放因子20221022v1.xlsx')
    sheet_list = pd.ExcelFile(ef_path).sheet_names
    for s in sheet_list:
        df_efp = pd.read_excel(ef_path, header=1, sheet_name=s)
        df_efp.columns = ['Province', 2011]
        process(df_efp, df_pp, s, df_c)

    df_coal = pd.DataFrame()
    for i in range(1, 7):
        temp = pd.read_csv(os.path.join(raw_path, '煤炭', 'result_%s.csv' % i))
        # 行转列
        temp = temp.set_index(['date']).stack().reset_index().rename(columns={'level_1': 'province', 0: 'data'})
        temp['data'] = temp['data'] / 1000000
        temp['sector'] = i
        df_coal = pd.concat([df_coal, temp]).reset_index(drop=True)
    province_list = df_coal['province'].unique()

    df_final = pd.DataFrame()
    for i in range(1, 7):
        for p in province_list:
            temp = df_coal[(df_coal['province'] == p) & (df_coal['sector'] == i)].reset_index(drop=True)
            df_fix = temp[temp['date'] >= '2022-01-01'].reset_index(drop=True)
            df_rest = temp[temp['date'] < '2022-01-01'].reset_index(drop=True)
            df_fix = df_fix.sort_values('date').reset_index(drop=True)
            df_fix = af.detect_outlier(df_fix, 'data', 'date')
            df_result = pd.concat([df_fix, df_rest]).reset_index(drop=True)
            df_final = pd.concat([df_final, df_result]).reset_index(drop=True)

    # 整理并输出
    df_final['department'] = 'Coal mining'
    df_final = pd.merge(df_final, df_c, left_on='province', right_on='中文')[
        ['date', 'data', '拼音', 'sector', 'department']].rename(columns={'拼音': 'province', 'data': 'value'})
    # 分三种情况
    # df_final['type'] = '不带回收'
    # df_final.to_csv(os.path.join(out_path, '煤炭', '煤炭_不带回收.csv'), index=False, encoding='utf_8_sig')
    # # 乘以90%
    # df_final['value'] = df_final['value'] * 0.9
    # df_final['type'] = '带回收_90%'
    # df_final.to_csv(os.path.join(out_path, '煤炭', '煤炭_带回收_90%.csv'), index=False, encoding='utf_8_sig')
    df_final['type'] = '带回收'
    df_final.to_csv(os.path.join(out_path, '煤炭', '煤炭_带回收.csv'), index=False, encoding='utf_8_sig')


if __name__ == '__main__':
    main()
