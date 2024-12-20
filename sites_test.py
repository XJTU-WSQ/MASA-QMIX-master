# sites_test.py
import unittest
import numpy as np
from utils.sites import Sites


class TestSites(unittest.TestCase):
    def setUp(self):
        self.sites = Sites()

    def test_rooms_pos_length(self):
        """测试老人的房间位置数量是否正确"""
        self.assertEqual(len(self.sites.rooms_pos), 18, "老人的房间数量应为18")

    def test_public_sites_pos_length(self):
        """测试公共活动区域位置数量是否正确"""
        self.assertEqual(len(self.sites.public_sites_pos), 8, "公共活动区域数量应为8")

    def test_sites_pos_concatenation(self):
        """测试任务请求点和任务目标点位置的拼接是否正确"""
        expected_num_sites = self.sites.n_rooms + self.sites.n_public_sites
        self.assertEqual(len(self.sites.sites_pos), expected_num_sites, "总任务点数量应等于房间数加公共区域数")


if __name__ == '__main__':
    unittest.main()