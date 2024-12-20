# robots_test.py
import unittest
import numpy as np
from utils.robots import Robots
from utils.sites import Sites


class TestRobots(unittest.TestCase):
    def setUp(self):
        self.robots = Robots()
        self.sites = Sites()

    def test_robot_initialization(self):
        """测试机器人初始化是否正确"""
        self.assertEqual(len(self.robots.robot_info), 5, "机器人类型数量应为5")
        self.assertEqual(len(self.robots.speed), 5, "机器人速度数组长度应为5")
        self.assertEqual(len(self.robots.robots_type_id), self.robots.num_robots, "机器人类型ID列表长度应等于机器人总数")
        self.assertEqual(self.robots.robot_pos.shape, (self.robots.num_robots, 2), "机器人位置应为(num_robots, 2)的数组")

    def test_get_skills(self):
        """测试获取机器人技能集是否正确"""
        for robot_id in range(self.robots.num_robots):
            expected_skills = self.robots.get_skills(robot_id)
            np.testing.assert_array_equal(self.robots.robots_skills[robot_id], expected_skills, f"Robot {robot_id} 的技能集不正确")

    def test_execute_task(self):
        """测试执行任务的方法"""
        robot_id = 0
        task = [0, 100, 1, 2, 3, 30]  # 示例任务
        time_on_road_1, total_time = self.robots.execute_task(robot_id, task)
        # 检查 robots_tasks_info 是否被正确更新
        np.testing.assert_array_equal(self.robots.robots_tasks_info[robot_id], [0, 100, 1, 2, 3], "任务信息未正确分配")
        # 检查 robots_time 是否被正确更新
        self.assertEqual(len(self.robots.robots_time[robot_id]), 3, "robots_time 应该有三个阶段")
        # 检查总时间是否合理
        self.assertTrue(total_time >= self.robots.robots_time[robot_id].sum(), "总时间应大于或等于各阶段时间之和")

    def test_renew_position(self):
        """测试更新机器人位置的方法"""
        robot_id = 0
        task_id = 2  # 送餐任务
        site_id = 1
        destination_id = 3
        time_span = 5  # 假设时间跨度为5分钟
        initial_pos = self.robots.robot_pos[robot_id].copy()
        self.robots.renew_position(robot_id, task_id, site_id, destination_id, time_span)
        # 检查位置是否更新
        self.assertNotEqual(list(initial_pos), list(self.robots.robot_pos[robot_id]), "机器人位置应被更新")


if __name__ == '__main__':
    unittest.main()