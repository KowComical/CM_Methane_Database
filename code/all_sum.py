import pandas as pd
import os
import global_function as af

global_path, raw_path, tools_path, out_path, df_c, end_year, start_year = af.base_element()


def main():
    all_sum()


def all_sum():
    all_files = af.search_file(out_path)
    df_result = pd.DataFrame()
    for file in all_files:
        if '部门' not in file:
            temp = pd.read_csv(file)
            df_result = pd.concat([df_result, temp]).reset_index(drop=True)

    # 全部门
    df_result = df_result[df_result['date'] >= f"{start_year}-01-01"].reset_index(drop=True)
    df_result.to_csv(os.path.join(out_path, '全部门', '全部门.csv'), index=False, encoding='utf_8_sig')
    # 分部门
    bumen = df_result['department'].unique()
    with pd.ExcelWriter(os.path.join(out_path, '全部门', '分部门.xlsx')) as writer:
        for b in bumen:
            temp = df_result[df_result['department'] == b].reset_index(drop=True)
            temp = temp.sort_values('date').reset_index(drop=True)
            temp.to_excel(writer, sheet_name=str(b), index=False)


if __name__ == '__main__':
    main()
