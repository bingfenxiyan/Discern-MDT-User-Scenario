import os
import datetime
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.utils import shuffle
from tqdm import tqdm


class Function:
    """公共函数"""
    def __init__(self):
        pass

    @staticmethod
    def timestamp(t):
        date = datetime.datetime.fromtimestamp(int(str(t)[:10]))
        return date

    @staticmethod
    def get_distance(lon_a, lat_a, lon_b, lat_b):
        """计算两个经纬度点距离"""
        radLonA, radLatA, radLonB, radLatB = map(np.deg2rad, [lon_a, lat_a, lon_b, lat_b])
        dlat = radLatB - radLatA
        dlon = radLonB - radLonA
        m = np.sin(dlat / 2) ** 2 + np.cos(radLatA) * np.cos(radLatB) * np.sin(dlon / 2) ** 2
        dis = 2 * np.arcsin(np.sqrt(m)) * 6378.137 * 1000
        return dis

    def multi_distance(self, lon, lat, by=None):
        """一个对多个经纬度返回最远距离"""
        dis = 0
        for once in by:
            ondis = self.get_distance(once[0], once[1], lon, lat)
            if ondis > dis:
                dis = ondis
        return dis

    @staticmethod
    def transform(data):
        """将ta和aoa转为标准数据"""
        data["intaoa"] = data["intaoa"].apply(lambda x: 0 if x is np.nan else x / 2 + 0.5)
        data["intmr_ta"] = data["intmr_ta"].apply(lambda x: 0 if x is np.nan else x * 78.12)
        return data


# --------数据处理------- #
class DislodgeSpeedAbnormal:
    """去除速度异常的数据"""
    def __init__(self):
        self.get_function = Function()

    def _speed_detection(self, data, key, ts, lon, lat, speed):
        data = data.reset_index(drop=True)
        data.sort_values(by=[key, ts], inplace=True)      # 排序，一级排序 key，二级排序 ts
        arr = np.array(data[[key, ts, lon, lat]])
        length = arr.shape[0]
        label = [True]
        speed_list = [0]
        for i in range(1, length):
            time_diff = (arr[i][1] - arr[i-1][1]).total_seconds()
            if (arr[i][0] == arr[i-1][0]) & (time_diff != 0):
                lon_a, lat_a, lon_b, lat_b = arr[i][2], arr[i][3], arr[i-1][2], arr[i-1][3]
                distance = self.get_function.get_distance(lon_a, lat_a, lon_b, lat_b)
                today_speed = (distance / time_diff) * 3.6
                if today_speed > speed:
                    label.append(False)
                else:
                    label.append(True)
                speed_list.append(today_speed)
            else:
                label.append(True)
                speed_list.append(0)
        data["速度"] = speed_list
        data = data[label]
        proportion = label.count(False) / len(label)
        return data, proportion

    def begin(self, data, key, ts="", lon="", lat="", speed=None, verify=None):
        """
        开始
        :param data: 数据, DataFrame
        :param key: 筛选数据用, str
        :param ts: 时间列, str
        :param lon: 经度列, str
        :param lat: 纬度列, str
        :param speed: 异常速度门限, int
        :param verify: 验证次数, int
        :return: data(DataFrame)
        """
        if verify is None or verify < 1:
            raise ValueError("传入值错误 verify=%s" % verify)
        data[ts] = data[ts].apply(lambda row: datetime.datetime.fromtimestamp(int(str(row)[:10])))
        for g in range(verify):
            data, pro = self._speed_detection(data, key, ts, lon, lat, speed)
            print("数据数量: %d,用户移动速度异常数据占比: %.3f%%" % (data.shape[0], pro * 100))
            if pro == 0:
                break
        return data


class DislodgeLocationAbnormal:
    """经纬度定位异常，超出站点覆盖范围"""
    def __init__(self):
        self.get_function = Function()

    def _location_detection(self, data, by=None, on=None, lon="", lat=""):
        if by is None:
            return data
        arr = np.array(data[[on, lon, lat]])
        label = []
        location_list = []
        for ser in arr:
            # 验证站点是否输入相同经纬度
            l1 = set()
            l2 = set()
            for lm in by[ser[0]]:
                l1.add(lm[0])
                l2.add(lm[1])
            # 区分一个或者多个经纬度计算
            if len(l1) == 1 and len(l2) == 1:
                dis = self.get_function.get_distance(ser[1], ser[2], by[ser[0]][0][0], by[ser[0]][0][1])
            else:
                dis = self.get_function.multi_distance(ser[1], ser[2], by[ser[0]])
            if len(by[ser[0]]) != 0:
                if dis > by[ser[0]][0][2]:
                    label.append(False)
                else:
                    label.append(True)
            else:
                label.append(True)
            location_list.append(dis)
        data["距站点距离"] = location_list
        data = data[label]
        proportion = label.count(False) / len(label)
        return data, proportion

    def begin(self, data, worker_data, on=None, left_on=None, right_on=None, lon_a="", lat_a="", lon_b="", lat_b="", site=None, scope=None):
        """
        开始
        :param data: 数据, DataFrame
        :param worker_data: 5G站点工参, DataFrame
        :param on: 连接列
        :param left_on: 左边连接列
        :param right_on: 右边连接列
        :param lon_a: data表经度列
        :param lat_a: data表纬度列
        :param lon_b: 工参表经度列
        :param lat_b: 工参表纬度列
        :param site: 如果有分站点类型,则为站点类型列
        :param scope: site=None时5G站点覆盖范围int, 或是 {"class1": scope1, "class2":scope2, ...}
        :return: data(DataFrame)
        """
        if on is not None and left_on is None and right_on is None:
            left_on, right_on = on, on
        on_list = data[left_on].drop_duplicates().values.tolist()
        storage_dict = {}
        if site is None and isinstance(scope, int):
            for key in on_list:
                unit_worker = worker_data.loc[worker_data[right_on] == key]
                storage_dict[key] = np.array([unit_worker[lon_b], unit_worker[lat_b], scope]).T
        elif site is not None and isinstance(scope, dict):
            for key in on_list:
                unit_worker = worker_data.loc[worker_data[right_on] == key]
                tt = [scope[x] for x in unit_worker[site].values.tolist()]
                storage_dict[key] = np.array([unit_worker[lon_b].values, unit_worker[lat_b].values, tt]).T
        else:
            raise ValueError("传入值错误 site=%s, scope=%s" % (site, scope))
        data, pro = self._location_detection(data, storage_dict, left_on, lon_a, lat_a)
        print("数据数量: %d, 经纬度定位位置异常数据占比: %.3f%%" % (data.shape[0], pro * 100))
        return data


class DislodgeAbnormal:
    """去除异常数据"""
    def __init__(self):
        pass

    def read(self, file, chunkSize=50000, patitions=10 ** 2, header=0, skiprows=None, usecols=None, encoding='gbk', engine="python"):
        """
        读取大文件
        :param file: 文件路径, str
        :param chunkSize: 读取大小, int
        :param patitions: 进度条设置, int
        :param skiprows: 跳过的行数, int
        :param usecols: 读取的列数, list
        :param encoding: 编码, str
        :return: 文档数据
        """
        format = os.path.split(file)[1].split('.')[1]
        if format == "csv":
            reader = pd.read_csv(file, iterator=True, engine=engine, header=header, skiprows=skiprows,
                                 usecols=usecols, encoding=encoding)
            chunks = []
            with tqdm(range(patitions), 'Reading') as t:
                for _ in t:
                    try:
                        chunk = reader.get_chunk(chunkSize)
                        chunks.append(chunk)
                    except StopIteration:
                        break
            data = pd.concat(chunks, ignore_index=True)
        elif format in ["xls", "xlsx"]:
            data = pd.read_excel(file, skiprows=skiprows, header=header, usecols=usecols, encoding=encoding)
        else:
            raise FileExistsError("The file type cannot be identified %s" % file)
        return data

    def run(self, file, header=0, skiprows=None, usecols=None, encoding='gbk', engine='python'):
        df = self.read(file, header=header, skiprows=skiprows,
                       usecols=usecols, encoding=encoding, engine=engine)
        columns = ['intstarttime', 'vcimsi', 'intenbid', 'intscrsrp','intnc1rsrp',
                   'intnc2rsrp', 'intnc3rsrp', 'intnc4rsrp', 'intnc5rsrp',
                   'intnc6rsrp', 'intnc7rsrp', 'intnc8rsrp', 'intmr_ta',
                   'intmr_uetranspowermargin', 'flong', 'flat']
        if "场景" in df.columns.values.tolist():
            columns.append("场景")
        df = df[columns]
        # 筛选出经纬度不为空的数据
        df = df.loc[(df["flong"] > 20) & (df["flat"] > 10)]
        df = df.dropna(subset=["flong", "flat"])
        # 去除移动速度异常数据
        df = DislodgeSpeedAbnormal().begin(df,
                                           key="vcimsi",
                                           ts="intstarttime",
                                           lon="flong",
                                           lat="flat",
                                           speed=120,
                                           verify=1)
        # 去除经纬度异常数据
        worker_data = pd.read_excel("./data/广州区域开启MDT区域小区表.xlsx", encoding="gbk")
        df = DislodgeLocationAbnormal().begin(df, worker_data,
                                              left_on="intenbid",
                                              right_on="ENODEBID",
                                              lon_a="flong",
                                              lat_a="flat",
                                              lon_b="LONB",
                                              lat_b="LATB",
                                              site="覆盖类型",
                                              scope={"室分": 300, "宏站": 800})
        df.to_csv("AbnormalEliminateData.csv", index=False, encoding="gbk")
        return df
# --------end------- #


# --------场景区分------- #
class SceneJudge:
    """场景区分"""
    def __init__(self):
        self.get_function = Function()
        self.dislodge_abnormal = DislodgeAbnormal()

    def run(self, k_list=[], ratio=0.3, train_path="", classify_path=""):
        """
        训练模型, 找到最佳k值, 使用最佳k值进行分类
        :param k_list: 范围,list
        :param ratio: 交叉验证占比
        :param train_path: 交叉验证数据集路径
        :param classify_path: 需要分类的数据集路径
        :return: 分类结果
        """
        train_set = self.dislodge_abnormal.run(train_path)
        # 去除没有邻区的
        train_set.dropna(subset=["intnc1rsrp"], inplace=True)
        train_set.replace(r"\N", np.nan, inplace=True)
        # 最佳k值
        k = self.cross_validation(k_list, ratio, train_set)
        # 模型建立
        neigh = Knn(n_neighbors=k, algorithm='auto')
        train_data, train_label, _, _, _, _ = self._cross_data(train_set, ratio=1)
        neigh.fit(train_data, train_label)

        # 读取分类数据
        classify_set = self.dislodge_abnormal.run(classify_path)
        classify_set.replace(r"\N", np.nan, inplace=True)
        # 开始分类
        classify_data = self._get_data(classify_set)
        classification_result = neigh.predict(classify_data)
        classify_set["场景"] = classification_result
        classify_set.to_csv("结果.csv", encoding="gbk", index=False)
        return classify_set

    def cross_validation(self, k_list=[], ratio=0.3, data=""):
        """
        训练模型, 获取最佳k值
        :param k_list: 范围,list
        :param ratio: 交叉验证占比
        :param data: 交叉验证数据集
        :return: 最佳k值
        """
        training_result = []
        for k in k_list:
            # 构建分类器
            neigh = Knn(n_neighbors=k, algorithm='auto')
            precision_list = []
            # 交叉验证
            for i in range(5):
                # data输入数据(只能保护数值), label标签, set全部数据
                train_data, train_label, train_set, test_data, test_label, test_set = self._cross_data(data, ratio)
                neigh.fit(train_data, train_label)
                classification_result = neigh.predict(test_data)
                classification_result = (classification_result == test_label).tolist()
                precision = classification_result.count(True) / len(classification_result) * 100
                precision_list.append(precision)
                print("当前k={}, 准确率: {}%".format(k, precision))
            training_result.append([np.mean(precision_list), np.var(precision_list), k])
            print("当前k={}, 平均准确率:{}, 准确率方差:{}".format(k, np.mean(precision_list), np.var(precision_list)))
        training_result.sort(key=lambda row: (row[0], -row[1]), reverse=True)
        print("最佳k={}, 平均准确率:{}, 准确率方差:{}".format(training_result[0][2], training_result[0][0], training_result[0][1]))
        return training_result[0][2]

    def _uniformization(self, data):
        """数值归一化"""
        min_values = np.nanmin(data, axis=0)
        max_values = np.nanmax(data, axis=0)
        ranges = max_values - min_values
        m = data.shape[0]
        norm_data_set = data - np.tile(min_values, (m, 1))
        norm_data_set = norm_data_set / np.tile(ranges, (m, 1))
        # 控制最小值不为0
        norm_data_set[norm_data_set == 0] = 0.01
        return norm_data_set

    def _cross_data(self, data, ratio):
        """交叉验证数据集获取"""
        # data = transform(data)    # 转为标准数据
        data = shuffle(data)
        label_list = data["场景"].values.tolist()
        columns = ["intscrsrp", "距站点距离", "速度", "intnc1rsrp", "intnc2rsrp", "intnc3rsrp",
                   "intnc4rsrp", "intnc5rsrp", "intnc6rsrp", "intnc7rsrp", "intnc8rsrp",
                   "intmr_ta", "intmr_uetranspowermargin"]
        data_set = data[columns]
        data_set = data_set.astype(float)
        data_set = self._uniformization(data_set)  # array
        data_set = np.nan_to_num(data_set)
        length = int(len(label_list) * ratio)
        train_data = data_set[:length, :]
        train_label = label_list[:length]
        train_set = data[:length]
        test_data = data_set[length:, :]
        test_label = label_list[length:]
        test_set = data[length:]
        return train_data, train_label, train_set, test_data, test_label, test_set

    def _get_data(self, data):
        """交叉验证数据集获取"""
        # data = transform(data)    # 转为标准数据
        columns = ["intscrsrp", "距站点距离", "速度", "intnc1rsrp", "intnc2rsrp", "intnc3rsrp",
                   "intnc4rsrp", "intnc5rsrp", "intnc6rsrp", "intnc7rsrp", "intnc8rsrp",
                   "intmr_ta", "intmr_uetranspowermargin"]
        data_set = data[columns]
        data_set = data_set.astype(float)
        data_set = self._uniformization(data_set)  # array
        data_set = np.nan_to_num(data_set)
        return data_set
# --------end------- #


if __name__ == '__main__':
    sj = SceneJudge()
    sj.run(list(range(41, 44, 2)), 0.32, "./data/mdt基础数据.csv", "./data/5-20.csv")
