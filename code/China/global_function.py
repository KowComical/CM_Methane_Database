# 爬取国家统计局
def get_json(dv, vc, dr):
    import re
    import time
    import json
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager

    options = webdriver.ChromeOptions()
    options.add_experimental_option("excludeSwitches", ["enable-logging"])
    options.add_argument('--ignore-ssl-errors=yes')  # 这两条会解决页面显示不安全问题
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')

    dbcode_value = dv
    wds_value = '[{"wdcode":"zb","valuecode":"%s"}]' % vc
    dfwds_value = '[{"wdcode":"sj","valuecode": "%s"}]' % dr
    k1_value = str(int(time.time() * 1000))
    url = 'https://data.stats.gov.cn/easyquery.htm?m=QueryData&dbcode=%s&rowcode=reg&colcode=sj&wds=%s&dfwds=%s' \
          '&k1=%s&h=1' % (dbcode_value, wds_value, dfwds_value, k1_value)
    # 开始爬
    wd = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)  # 打开浏览器
    wd.get(url)
    # 得到网页源码
    content = wd.page_source
    # 提取有效数据
    json_data = re.compile(r'pre-wrap;">(?P<data>.*?)</pre>', re.S)
    result = json.loads(json_data.findall(content)[0])
    # 提取数据
    name = []
    date = []
    data = []
    city_list = pd.json_normalize(result['returndata']['wdnodes'][1], record_path='nodes')['name'].tolist()
    time_list = pd.json_normalize(result['returndata']['wdnodes'][2], record_path='nodes')['name'].tolist()
    for c in city_list:
        for t in time_list:
            name.append(c)
            date.append(t)
    data_list = result['returndata']['datanodes']
    for i in range(len(data_list)):
        data.append(pd.DataFrame([data_list[i]['data']])['data'].tolist()[0])
    df_result = pd.concat([pd.DataFrame(name, columns=['name']), pd.DataFrame(date, columns=['date']),
                           pd.DataFrame(data, columns=['data'])], axis=1)
    wd.quit()
    return df_result


# 找到当前路径下的所有文件 包含子文件夹
def search_file(file_path):
    import os
    import sys
    sys.dont_write_bytecode = True
    file_name = []
    for parent, surnames, filenames in os.walk(file_path):
        for fn in filenames:
            file_name.append(os.path.join(parent, fn))
    return file_name


# 预测年度数据
def predict_yearly(df):
    import pandas as pd
    pro_list = df['province'].unique()
    date_range = pd.date_range(start='2010', end='2023', freq='y').strftime('%Y').astype(int)
    df_all = pd.DataFrame()
    for p in pro_list:
        temp = df[df['province'] == p].reset_index(drop=True)
        for d in date_range:
            if d not in temp['date'].tolist():
                df_date = pd.DataFrame([d], columns=['date'])
                temp = pd.concat([temp, df_date])
            df_temp = temp.sort_values('date').reset_index(drop=True)
            df_temp['province'] = p
            df_temp = df_temp.set_index('date')
            df_temp['data'] = df_temp['data'].interpolate(method='linear').ffill().bfill()
            df_temp = df_temp.reset_index()
            df_all = pd.concat([df_all, df_temp]).reset_index(drop=True)
    df_all = df_all.groupby(['date', 'province']).mean().reset_index()
    df_all['date'] = df_all['date'].astype(int)
    return df_all


# 线性预测年度数据并拆分到月
def linear_predict(df, end_year):
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    province_list = df['province'].unique()
    df_final = pd.DataFrame()
    for pro in province_list:
        test = df[df['province'] == pro].reset_index(drop=True)
        test = test.set_index('date').resample('Y').sum(numeric_only=True).reset_index()
        test['date'] = test['date'].dt.year
        # 行转列
        test = test.set_index(['date']).stack().reset_index().rename(columns={'level_1': 'type', 0: 'value'})
        type_list = test['type'].unique()
        df_rest = pd.DataFrame()
        df_not_all = pd.DataFrame()
        for p in type_list:
            temp = test[test['type'] == p].reset_index(drop=True)
            # 前几年突然消失数据的一并外推
            if len(temp[temp['value'] == 0]) != len(temp):
                # 得到所有为0的年份
                new_temp = temp[temp['value'] != 0].reset_index(drop=True)
                start_year = max(new_temp['date']) + 1
                df_date = pd.DataFrame()
                for i in range(start_year, end_year):
                    X = new_temp['date'].values.reshape(-1, 1)
                    y = new_temp['value'].values.reshape(-1, 1)

                    model = LinearRegression()
                    model.fit(X, y)

                    X_predict = pd.DataFrame([i]).values.reshape(-1, 1)
                    y_predict = model.predict(X_predict)
                    predict = pd.DataFrame([[int(X_predict), float(y_predict)]], columns=['date', 'value'])
                    df_date = pd.concat([df_date, predict]).reset_index(drop=True)
                    df_date['type'] = p
                    new_temp = pd.concat([new_temp, df_date]).reset_index(drop=True)
                df_not_all = pd.concat([df_not_all, new_temp]).reset_index(drop=True)
            else:
                start_year = max(temp['date']) + 1
                for i in range(start_year, start_year + 1):
                    X = temp['date'].values.reshape(-1, 1)
                    y = temp['value'].values.reshape(-1, 1)

                    model = LinearRegression()
                    model.fit(X, y)

                    X_predict = pd.DataFrame([i]).values.reshape(-1, 1)
                    y_predict = model.predict(X_predict)

                    # 将结果加入df中
                    x_predict = pd.DataFrame([[int(X_predict), float(y_predict)]], columns=['date', 'value'])
                    x_predict['type'] = p
                    temp = pd.concat([temp, x_predict]).reset_index(drop=True)
                    df_rest = pd.concat([df_rest, temp]).reset_index(drop=True)

        df_predict = pd.concat([df_rest, df_not_all]).reset_index(drop=True)
        df_predict = df_predict.rename(columns={'value': 'sum', 'date': 'year'})

        data_df_new = df[df['province'] == pro].reset_index(drop=True)
        missing_date = pd.to_datetime(
            pd.date_range(start='%s-01-01' % start_year, end='%s-01-01' % end_year, freq='M').strftime('%Y-%m'))
        missing_date = pd.DataFrame(missing_date, columns=['date'])
        missing_date['province'] = pro
        data_df_new = pd.concat([data_df_new, missing_date]).reset_index(drop=True).fillna(0)
        data_df_new = data_df_new.set_index(['date', 'province']).stack().reset_index().rename(
            columns={'level_2': 'type', 0: 'value'})
        data_df_new['year'] = data_df_new['date'].dt.year
        # 计算占比
        df_ratio = pd.merge(data_df_new, df_predict)
        df_ratio['ratio'] = df_ratio['value'] / df_ratio['sum']
        df_ratio['month_date'] = df_ratio['date'].dt.strftime('%m-%d')

        # 列转行
        df_ratio = pd.pivot_table(df_ratio.fillna(0), index=['month_date', 'province', 'year'], values='ratio',
                                  columns='type').reset_index().replace(0, np.nan).ffill().fillna(0)
        # 行转列
        df_ratio = df_ratio.set_index(['province', 'month_date', 'year']).stack().reset_index().rename(
            columns={'level_3': 'type', 0: 'ratio'})
        # 拆分到月
        df_new_result = pd.merge(df_predict, df_ratio)
        df_new_result['value'] = df_new_result['sum'] * df_new_result['ratio']
        df_new_result['date'] = pd.to_datetime(df_new_result['year'].astype(str) + '-' + df_new_result['month_date'])
        df_new_result = df_new_result[~df_new_result.duplicated()].reset_index(drop=True)
        df_final = pd.concat([df_final, df_new_result]).reset_index(drop=True)
    return df_final


# 所有基础信息
def base_element():
    import os
    import pandas as pd
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
    return global_path, raw_path, tools_path, out_path, df_c, end_year, start_year


# 监测并覆盖异常值
def detect_outlier(df_result, value, date):
    import numpy as np
    import pandas as pd
    n = 2.6  # n*sigma

    data_y = df_result[value]
    data_x = df_result[date]

    ymean = np.mean(data_y)
    ystd = np.std(data_y)
    threshold1 = ymean - n * ystd
    threshold2 = ymean + n * ystd

    outlier = []  # 将异常值保存
    outlier_x = []

    for i in range(len(data_y)):
        if (data_y[i] < threshold1) | (data_y[i] > threshold2):
            outlier.append(data_y[i])
            outlier_x.append(data_x[i])
        else:
            continue
    # 将异常值转为null然后按照均值填充
    df_null = df_result[df_result[date].isin(outlier_x)].reset_index(drop=True)
    df_null[value] = np.nan
    df_rest = df_result[~df_result[date].isin(outlier_x)].reset_index(drop=True)
    df_result = pd.concat([df_rest, df_null]).reset_index(drop=True)
    df_result[value] = df_result[value].interpolate()
    df_result = df_result.sort_values(date).reset_index(drop=True)
    return df_result


# 监测一个list里是否都是一样的值
def all_same(lst):
    return all(x == lst[0] for x in lst)


# 重新分配月度
def redistribute_monthly_values(original_values, target_values):
    import numpy as np
    original_sum = np.sum(original_values)
    target_sum = np.sum(target_values)
    redistribution_factor = target_sum / original_sum
    redistributed_values = original_values * redistribution_factor
    return list(redistributed_values)
