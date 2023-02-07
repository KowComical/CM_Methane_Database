import 固体废弃物 as solid_waste
import 家畜 as live_stock
import 废水 as waste_water
import 水稻 as rice_cultivation
import 油气 as oil_gas
import 湿地 as wet_land
import 能源燃烧 as energy
import 煤炭 as coal
import all_sum as sum_all

from contextlib import contextmanager
import time
import re
start_year = 2013


@contextmanager
# 设置一个可以监测每个function运行的时间的function
def timeit_context(name):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    print('##### [{}] finished in {:.2f} minutes #####'.format(name, elapsed_time / 60))


# from contextlib import suppress
func_list = [solid_waste, live_stock, waste_water, rice_cultivation, oil_gas, wet_land, energy, coal, sum_all]
# func_list = [solid_waste]
for my_func in func_list:
    # with suppress(Exception):  # 如果出错则pass（并不报错） 这里并不用这种办法
    function_name = re.findall(r"([^\\/]*).....$", str(my_func))[0]
    with timeit_context(function_name):
        try:
            my_func.main()
        except Exception as e:
            print(e)
