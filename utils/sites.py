"""
记录老人房间位置以及公共活动区域的位置
A:老人房间: 分别对应老人的1-18号房间
B:公共活动区域: 餐厅，室外活动区南，室外活动区北，活动室，浴室，值班室，西北卫生间，东南卫生间。
"""
import numpy as np


# Sites里面存储的是任务请求点和任务目标点的位置信息
class Sites:
    def __init__(self):
        # 老人房间: 分别对应老人的1-18号房间
        self.rooms_pos = np.array([
            [10, 5], [20, 5], [30, 5], [40, 5], [50, 5], [60, 5], [70, 5], [80, 5], [90, 5],
            [20, 50], [30, 50], [40, 50], [50, 50], [60, 50], [70, 50], [80, 50], [90, 50], [100, 50]
        ])
        # 公共活动区域: 餐厅，室外活动区南，室外活动区北，活动室，浴室，医护值班室，西北卫生间，东南卫生间。
        self.public_sites_pos = np.array([
            [28, 25], [60, 10], [60, 40], [90, 25], [104, 35], [104, 15], [20, 50], [110, 5]
        ])
        # 养老机构中所有任务请求点和任务目标点位置：
        self.sites_pos = np.concatenate((self.rooms_pos, self.public_sites_pos), axis=0)
        self.n_rooms = len(self.rooms_pos)
        self.n_public_sites = len(self.public_sites_pos)
        self.num_sites = self.n_public_sites + self.n_rooms
