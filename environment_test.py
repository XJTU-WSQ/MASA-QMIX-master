# environment_test.py
import unittest
import numpy as np
from environment import ScheduleEnv


class TestScheduleEnv(unittest.TestCase):
    def setUp(self):
        self.env = ScheduleEnv()
        self.env.reset()

    def test_reset(self):
        """测试环境重置是否正确"""
        self.env.reset()
        self.assertEqual(self.env.time, 0.0, "时间应被重置为0")
        self.assertFalse(self.env.done, "环境完成标志应为False")
        self.assertEqual(self.env.step_count, 0, "步数应被重置为0")
        self.assertEqual(sum(self.env.robots_state), 0, "所有机器人应为空闲状态")

    def test_get_state(self):
        """测试获取环境状态是否正确"""
        state = self.env.get_state()
        self.assertEqual(len(state), 17, "状态向量长度应为17")

    def test_get_avail_actions(self):
        """测试获取可用动作是否正确"""
        agent_id = 0
        avail_actions = self.env.get_avail_actions(agent_id)
        self.assertEqual(len(avail_actions), 8, "可用动作长度应为8")
        self.assertIn(0, avail_actions, "可用动作应包含0")
        self.assertIn(1, avail_actions, "可用动作应包含1")

    def test_step(self):
        """测试环境步进是否正确"""
        # 选择所有机器人都不执行任何动作
        actions = [7 for _ in range(self.env.robots.num_robots)]
        reward, done = self.env.step(actions)
        self.assertIsInstance(reward, float, "奖励应为浮点数")
        self.assertIsInstance(done, bool, "完成标志应为布尔值")

    def test_choose_sd_action(self):
        """测试选择动作的方法是否返回可用动作中的一个"""
        agent_id = 0
        avail_action = [1, 0, 1, 0, 0, 0, 0, 1]
        chosen_action = self.env.choose_sd_action([avail_action], agent_id)
        self.assertIn(chosen_action, [0, 2, 7], "选择的动作应在可用动作中")

    def test_reward_function(self):
        """测试奖励函数是否正确"""
        wait_time = 10
        time_on_road = 5
        expected_reward = -(wait_time + time_on_road) * 0.01
        reward = self.env.reward_function(wait_time, time_on_road)
        self.assertEqual(reward, expected_reward, "奖励应与预期相同")


if __name__ == '__main__':
    unittest.main()