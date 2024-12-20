# tasks_test.py
import unittest
import numpy as np
from utils.tasks import Tasks


class TestTasks(unittest.TestCase):
    def setUp(self):
        self.tasks = Tasks()

    def test_task_info_length(self):
        """测试任务信息的长度是否正确"""
        self.assertEqual(len(self.tasks.task_info), 7, "任务类型数量应为7")

    def test_required_skills_length(self):
        """测试每个任务的技能要求长度是否为6"""
        for skill in self.tasks.required_skills:
            self.assertEqual(len(skill), 6, "每个任务的技能要求长度应为6")

    def test_task_priority_length(self):
        """测试任务优先级列表的长度是否正确"""
        self.assertEqual(len(self.tasks.task_priority), 7, "任务优先级数量应为7")

    def test_get_tasks_required_skill_list(self):
        """测试获取任务窗口中任务的技能列表是否正确"""
        # 创建一个任务窗口，包含任务ID 1、2、3
        task_window = [
            [0, 1, 2, 3, 0, 0],  # 假设任务ID为0代表紧急事件
            [0, 1, 0, 1, 0, 1],  # 任务ID为1
            [0, 0, 1, 2, 0, 0]  # 任务ID为2
        ]
        expected_skills = [
            self.tasks.required_skills[0],
            self.tasks.required_skills[1],
            self.tasks.required_skills[2]
        ]
        result = self.tasks.get_tasks_required_skill_list(task_window)
        np.testing.assert_array_equal(result, expected_skills, "技能列表应与预期相同")


if __name__ == '__main__':
    unittest.main()

