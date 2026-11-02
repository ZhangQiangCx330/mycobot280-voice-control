"""
MyCobot 280 MuJoCo 3D 仿真
加载官方 myCobot 280 Jetson Nano 真实 STL 网格模型，渲染质量优于 PyBullet 圆柱体。
"""

import os
import math
import time

import mujoco
import mujoco.viewer


class MyCobotMujoco:
    """MyCobot 280 MuJoCo 仿真封装

    关节索引与 pymycobot 一致：
      qpos[0] = J1 底座   qpos[1] = J2 大臂   qpos[2] = J3 小臂
      qpos[3] = J4 腕部   qpos[4] = J5 末端   qpos[5] = J6 旋转
    """

    MODEL_XML = os.path.join(os.path.dirname(__file__), "xml", "mycobot_280jn_mujoco.xml")

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(self.MODEL_XML)
        self.data = mujoco.MjData(self.model)

        # 将所有部件颜色设为白色
        for gid in range(self.model.ngeom):
            self.model.geom_rgba[gid] = [0.95, 0.95, 0.95, 1.0]

        mujoco.mj_forward(self.model, self.data)

        # 打开渲染窗口（后台线程运行）
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

        # 初始归零
        self._set_qpos([0.0] * 6)

    def _set_qpos(self, angles_rad):
        """直接设置 6 个关节角度（弧度）"""
        for i in range(6):
            self.data.qpos[i] = angles_rad[i]
        mujoco.mj_forward(self.model, self.data)

    def send_angles(self, angles_deg, speed=30):
        """设置目标角度（度），带插值动画"""
        start = [self.data.qpos[i] for i in range(6)]
        target = [math.radians(a) for a in angles_deg]

        steps = 60
        delay = max(0.002, 0.012 - speed * 0.0001)
        for s in range(1, steps + 1):
            t = s / steps
            interp = [start[i] + (target[i] - start[i]) * t for i in range(6)]
            self._set_qpos(interp)
            self.viewer.sync()
            time.sleep(delay)

        self._set_qpos(target)
        self.viewer.sync()

    def send_angle(self, joint_id, angle_deg, speed=30):
        """单独控制某个关节（1~6）"""
        angles = list(self.get_angles())
        if 1 <= joint_id <= 6:
            angles[joint_id - 1] = angle_deg
        self.send_angles(angles, speed)

    def get_angles(self):
        """获取当前关节角度（度）"""
        return [round(math.degrees(self.data.qpos[i]), 1) for i in range(6)]

    def release_all(self):
        print("  Release all")

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    print("=" * 50)
    print("MyCobot 280 MuJoCo Simulation")
    print("=" * 50)

    robot = MyCobotMujoco()

    print("\n[1] 归零")
    robot.send_angles([0, 0, 0, 0, 0, 0], 30)
    time.sleep(1)

    print("\n[2] 抬大臂")
    robot.send_angles([0, -30, 0, 0, 0, 0], 30)
    time.sleep(1)

    print("\n[3] 弯小臂")
    robot.send_angles([0, -30, -45, 0, 0, 0], 30)
    time.sleep(1)

    print("\n[4] 转底座")
    robot.send_angles([45, -30, -45, 0, 0, 0], 30)
    time.sleep(1)

    print("\n[5] 回原点")
    robot.send_angles([0, 0, 0, 0, 0, 0], 30)

    print("\nDone! 关闭 MuJoCo 窗口退出。")
    input("Press Enter to close...")
    robot.close()
