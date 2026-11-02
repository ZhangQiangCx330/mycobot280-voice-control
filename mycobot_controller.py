# 机械臂综合控制 — Socket 连接 + 全功能动作库 + MuJoCo 3D 仿真
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import json
import math
import threading
import socket

ROBOT_IP = "192.168.31.214"
ROBOT_PORT = 9000
TIMEOUT = 5

mc = None


# ==================== 动作中断机制 ====================

_interrupt_event = threading.Event()
_native_sleep = time.sleep


class ActionInterrupted(Exception):
    """动作被新指令打断"""


def request_interrupt():
    """请求打断当前正在执行的动作"""
    _interrupt_event.set()


def clear_interrupt():
    """清除中断标志（新动作开始前调用）"""
    _interrupt_event.clear()


def interruptible_sleep(seconds):
    """可被打断的等待，被中断时抛出 ActionInterrupted"""
    end = time.time() + seconds
    while time.time() < end:
        if _interrupt_event.is_set():
            raise ActionInterrupted()
        _native_sleep(0.05)


# ==================== 连接与异常处理 ====================

def connect_robot(ip=ROBOT_IP, port=ROBOT_PORT, timeout=TIMEOUT):
    """带异常处理的机械臂连接"""
    global mc
    from pymycobot import MyCobot280Socket

    print(f"[INFO] 正在连接机械臂 {ip}:{port} ...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)

    try:
        sock.connect((ip, port))
        sock.close()
        mc = MyCobot280Socket(ip, port)
        print(f"[OK] 成功连接机械臂 {ip}:{port}")
        angles = mc.get_angles()
        if angles:
            print(f"[OK] 当前角度: {angles}")
        return True
    except socket.timeout:
        print(f"[ERROR] 连接超时 — 请检查 IP 是否可 ping 通、机械臂是否上电")
        return False
    except ConnectionRefusedError:
        print(f"[ERROR] 连接被拒绝 — 请检查端口 {port} 是否正确")
        return False
    except socket.gaierror:
        print(f"[ERROR] 无法解析地址 — 请检查 IP 格式或网络是否连接")
        return False
    except OSError as e:
        print(f"[ERROR] 网络错误: {e}")
        return False


# ==================== 仿真模式 ====================

class SimRobot:
    """仿真机械臂，模拟全部 pymycobot API"""
    def __init__(self):
        self._angles = [0, 0, 0, 0, 0, 0]
        self._coords = [160, 0, 160, 0, 0, 0]
        self._gripper = 0
        self._color = (0, 0, 0)
        print("[SIM] 仿真模式已启动")

    def get_angles(self):
        return list(self._angles)

    def send_angles(self, angles, speed):
        self._angles = list(angles)
        print(f"[SIM] 角度 → {angles}  速度 {speed}")

    def send_angle(self, joint_id, angle, speed):
        if 1 <= joint_id <= 6:
            self._angles[joint_id - 1] = angle
        print(f"[SIM] 关节{joint_id} → {angle}°")

    def send_coords(self, coords, speed, mode=0):
        self._coords = list(coords)
        print(f"[SIM] 坐标 → {coords}")

    def get_coords(self):
        return list(self._coords)

    def set_color(self, r, g, b):
        self._color = (r, g, b)
        print(f"[SIM] LED → RGB({r},{g},{b})")

    def set_gripper_value(self, value, speed):
        self._gripper = max(0, min(100, value))
        print(f"[SIM] 夹爪 → {self._gripper}%")

    def set_gripper_state(self, state, speed):
        print(f"[SIM] 夹爪 → {'闭合' if state else '张开'}")

    def get_gripper_value(self):
        return self._gripper

    def is_gripper_moving(self):
        return False

    def is_in_position(self, angles, error=0):
        return True

    def release_all_servos(self):
        print("[SIM] 释放所有舵机")

    def set_encoder(self, joint_id, value, speed):
        print(f"[SIM] 编码器 关节{joint_id} → {value}")

    def set_encoders(self, values, speed):
        print(f"[SIM] 编码器 → {values}")

    def get_encoder(self, joint_id):
        return 2048

    def is_controller_connected(self):
        return 1


# ==================== MuJoCo 3D 仿真适配器 ====================

class Mujoco3DRobot:
    """MuJoCo 3D 仿真 — 加载官方 STL 网格模型，适配为完整 pymycobot API"""

    def __init__(self):
        from mycobot_mujoco import MyCobotMujoco
        print("[3D] 启动 MuJoCo 物理仿真...")
        self._robot = MyCobotMujoco()
        self._gripper = 0
        self._color = (0, 0, 0)
        print("[3D] 3D 仿真就绪（查看 MuJoCo 窗口）")

    def get_angles(self):
        return self._robot.get_angles()

    def send_angles(self, angles, speed):
        print(f"[3D] 角度 → {angles}  速度 {speed}")
        self._robot.send_angles(angles, speed)

    def send_angle(self, joint_id, angle, speed):
        print(f"[3D] 关节{joint_id} → {angle}°")
        self._robot.send_angle(joint_id, angle, speed)

    def send_coords(self, coords, speed, mode=0):
        print(f"[3D] 坐标 → {coords}（逆运动学暂未实现）")

    def get_coords(self):
        return [160, 0, 160, 0, 0, 0]

    def set_color(self, r, g, b):
        self._color = (r, g, b)
        print(f"[3D] LED → RGB({r},{g},{b})")

    def set_gripper_value(self, value, speed):
        self._gripper = max(0, min(100, value))
        print(f"[3D] 夹爪 → {self._gripper}%")

    def set_gripper_state(self, state, speed):
        self.set_gripper_value(100 if state else 0, speed)

    def get_gripper_value(self):
        return self._gripper

    def is_gripper_moving(self):
        return False

    def is_in_position(self, angles, error=0):
        current = self.get_angles()
        if not current:
            return True
        return all(abs(c - t) <= max(error, 3.0) for c, t in zip(current, angles))

    def release_all_servos(self):
        self._robot.release_all()

    def set_encoder(self, joint_id, value, speed):
        print(f"[3D] 编码器 关节{joint_id} → {value}")

    def set_encoders(self, values, speed):
        print(f"[3D] 编码器 → {values}")

    def get_encoder(self, joint_id):
        return 2048

    def is_controller_connected(self):
        return 1

    def close(self):
        self._robot.close()
        print("[3D] MuJoCo 窗口已关闭")


# ==================== 基础动作 ====================

def back_zero():
    print('机械臂归零')
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    interruptible_sleep(3)

def relax_arms():
    print('放松机械臂关节')
    mc.release_all_servos()

def emergency_stop():
    print('急停：立即停止所有动作')
    mc.release_all_servos()
    print('已急停，所有舵机已释放')

def single_joint_move(joint_index, angle):
    print(f'关节 {joint_index} 旋转至 {angle} 度')
    mc.send_angle(joint_index, angle, 40)
    interruptible_sleep(2)

def move_to_coords(X=150, Y=-130, HEIGHT_SAFE=230):
    print(f'移动至坐标：X {X} Y {Y}')
    mc.send_coords([X, Y, HEIGHT_SAFE, 0, 180, 90], 20, 0)
    interruptible_sleep(4)


# ==================== 姿态动作 ====================

def head_shake():
    print("左右摆头")
    mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)
    interruptible_sleep(1)
    for _ in range(2):
        mc.send_angle(5, 30, 80)
        interruptible_sleep(0.5)
        mc.send_angle(5, -30, 80)
        interruptible_sleep(0.5)
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    interruptible_sleep(2)

def head_nod():
    print("点头")
    mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)
    for _ in range(2):
        mc.send_angle(4, 13, 70)
        interruptible_sleep(0.5)
        mc.send_angle(4, -20, 70)
        interruptible_sleep(1)
        mc.send_angle(4, 13, 70)
        interruptible_sleep(0.5)
    mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)

def move_to_top_view():
    print('移动至俯视姿态')
    mc.send_angles([-62.13, 8.96, -87.71, -14.41, 2.54, -16.34], 10)
    interruptible_sleep(3)

def head_dance():
    print("头部舞蹈")
    mc.send_angles([0.87, -50.44, 47.28, 0.35, -0.43, -0.26], 70)
    interruptible_sleep(1)
    mc.send_angles([-0.17, -94.3, 118.91, -39.9, 59.32, -0.52], 80)
    mc.send_angles([-1.49, 115, -153.45, 30, -33.42, 137.9], 80)
    mc.set_color(0, 0, 50)
    interruptible_sleep(0.7)
    mc.send_angles([-1.49, 55, -153.45, 80, 33.42, 137.9], 80)
    mc.set_color(0, 50, 0)
    interruptible_sleep(1)
    mc.send_angles([0, 0, 0, 0, 0, 0], 80)


# ==================== 舞蹈动作 ====================

def move_to_pose(angles, speed=80, timeout=5, tolerance=5):
    start = time.time()
    mc.send_angles(angles, speed)
    interruptible_sleep(0.5)
    while not mc.is_in_position(angles, 0):
        if time.time() - start > timeout:
            print(f"移动超时（{timeout}s），继续")
            break
        interruptible_sleep(0.1)
    print(f"到达位置: {angles}")

def dance_performance(cycles=8):
    pose_A = [-30, 40, -50, 0, 30, 0]
    pose_B = [30, -40, 50, 0, -30, 0]
    print("舞蹈表演开始")
    for i in range(cycles):
        print(f"第 {i+1} 拍")
        move_to_pose(pose_A, 80)
        mc.set_color(50, 0, 100)
        interruptible_sleep(0.5)
        move_to_pose(pose_B, 80)
        mc.set_color(255, 50, 0)
        interruptible_sleep(0.5)
    move_to_pose([0, 0, 0, 0, 0, 0], 50)
    mc.set_color(0, 255, 255)
    interruptible_sleep(1)
    mc.set_color(0, 0, 0)
    print("表演完毕")

def dance_with_gripper(cycles=5):
    pose1 = [-30, 30, -45, 0, 30, 0]
    pose2 = [30, -30, 45, 0, -30, 0]
    print("舞蹈 + 夹爪")
    for i in range(cycles):
        print(f"第 {i+1} 拍")
        mc.send_angles(pose1, 80)
        mc.set_gripper_state(0, 50)
        mc.set_color(255, 0, 0)
        interruptible_sleep(1)
        mc.send_angles(pose2, 80)
        mc.set_gripper_state(1, 50)
        mc.set_color(0, 255, 0)
        interruptible_sleep(1)
    back_zero()
    mc.set_gripper_state(0, 50)
    mc.set_color(0, 0, 255)
    interruptible_sleep(2)
    mc.set_color(0, 0, 0)


# ==================== 夹爪与抓取 ====================

def gripper_test():
    print("夹爪功能测试")
    flag = mc.is_gripper_moving()
    print(f"夹爪是否在移动: {flag}")
    interruptible_sleep(0.5)
    mc.set_encoder(7, 2048, 20)
    interruptible_sleep(3)
    mc.set_encoder(7, 1300, 20)
    interruptible_sleep(3)
    mc.set_gripper_value(100, 70)
    interruptible_sleep(3)
    mc.set_gripper_value(0, 70)
    interruptible_sleep(3)
    for i in range(5):
        print(f"  第 {i+1} 次: 张开")
        mc.set_gripper_state(0, 70)
        interruptible_sleep(2)
        print(f"  第 {i+1} 次: 闭合")
        mc.set_gripper_state(1, 70)
        interruptible_sleep(2)
    print(f"当前夹爪值: {mc.get_gripper_value()}")

def pick_and_place_demo():
    print("抓取放置演示")
    mc.set_gripper_state(0, 50)
    mc.send_angles([0, 45, -45, 0, 0, 0], 50)
    interruptible_sleep(3)
    mc.set_gripper_state(1, 30)
    interruptible_sleep(3)
    mc.send_angles([0, 60, -60, 0, 0, 0], 50)
    interruptible_sleep(3)
    mc.send_angles([45, 45, -45, 0, 0, 0], 50)
    interruptible_sleep(3)
    mc.set_gripper_state(0, 50)
    interruptible_sleep(3)
    back_zero()
    print("抓取放置完成")


# ==================== RGB 灯光 ====================

def light_off():
    print("关灯")
    mc.set_color(0, 0, 0)


def rgb_cycle(cycles=7):
    print("RGB 灯循环展示")
    for i in range(cycles):
        mc.set_color(0, 0, 255)
        interruptible_sleep(0.5)
        mc.set_color(255, 0, 0)
        interruptible_sleep(0.5)
        mc.set_color(0, 255, 0)
        interruptible_sleep(0.5)
    mc.set_color(0, 0, 0)


# ==================== 多关节运动 ====================

def multi_joint_move():
    print("多关节联动")
    mc.send_angles([0, 0, 0, 0, 0, 0], 50)
    interruptible_sleep(2.5)
    mc.send_angles([90, 45, -90, 90, -90, 90], 50)
    interruptible_sleep(2.5)
    mc.send_angles([0, 0, 0, 0, 0, 0], 50)
    interruptible_sleep(2.5)
    mc.send_angles([-90, -45, 90, -90, 90, -90], 50)
    interruptible_sleep(2.5)
    back_zero()


# ==================== 示教录制 ====================

_record_data = []
_recording = False

def start_recording():
    global _recording, _record_data
    if _recording:
        print("已在录制中")
        return
    _record_data = []
    _recording = True
    print("开始录制...")
    def _loop():
        start = time.time()
        while _recording:
            angles = mc.get_angles()
            if angles:
                _record_data.append({'time': time.time() - start, 'angles': angles})
            interruptible_sleep(0.1)
    threading.Thread(target=_loop, daemon=True).start()

def stop_recording():
    global _recording
    _recording = False
    interruptible_sleep(0.3)
    if _record_data:
        print(f"停止，共 {len(_record_data)} 个采样点，{_record_data[-1]['time']:.1f}s")
    else:
        print("无数据")

def play_trajectory(speed=80):
    if not _record_data:
        print("无数据")
        return
    for i, pt in enumerate(_record_data):
        mc.send_angles(pt['angles'], speed)
        if i < len(_record_data) - 1:
            wait = _record_data[i+1]['time'] - pt['time']
            if wait > 0:
                interruptible_sleep(wait)
    print("回放完成")

def save_trajectory(filename="record.json"):
    if not _record_data:
        print("无数据")
        return
    with open(filename, 'w') as f:
        json.dump({'info': {'points': len(_record_data), 'date': time.strftime('%Y-%m-%d %H:%M:%S')}, 'data': _record_data}, f, indent=2)
    print(f"已保存到 {filename}")

def load_trajectory(filename="record.json"):
    global _record_data
    with open(filename, 'r') as f:
        loaded = json.load(f)
    _record_data = loaded.get('data', loaded) if isinstance(loaded, dict) else loaded
    print(f"已加载 {len(_record_data)} 个采样点")


# ==================== 创新动作 ====================

def draw_square(speed=50):
    corners = [
        [0, 30, -50, -30, 0, 0],
        [0, 30, -50, 30, 0, 0],
        [0, 50, -70, 30, 0, 0],
        [0, 50, -70, -30, 0, 0],
        [0, 30, -50, -30, 0, 0],
    ]
    for i, angles in enumerate(corners):
        mc.send_angles(angles, speed)
        mc.set_color(255, 60 * i, 0)
        interruptible_sleep(2)
    mc.set_color(0, 255, 0)
    interruptible_sleep(0.5)
    mc.set_color(0, 0, 0)
    back_zero()
    print("画正方形完成")

def breathing(cycles=5):
    opened = [0, 20, -30, 0, 30, 0]
    closed = [0, 60, -80, 0, 60, 0]
    for i in range(cycles):
        print(f"呼吸 {i+1} — 吸气")
        mc.send_angles(opened, 30)
        for b in range(0, 255, 5):
            mc.set_color(b, b // 2, b)
            interruptible_sleep(0.02)
        print(f"呼吸 {i+1} — 呼气")
        mc.send_angles(closed, 30)
        for b in range(255, 0, -5):
            mc.set_color(b, b // 2, b)
            interruptible_sleep(0.02)
    mc.set_color(0, 0, 0)
    back_zero()
    print("呼吸效果完成")


# ==================== 主菜单 ====================

def action_menu():
    menu = """
动作选择:
  1 - 归零              2 - 放松关节         3 - 单关节运动
  4 - 坐标移动          5 - 摆头             6 - 点头
  7 - 头部舞蹈          8 - 舞蹈表演         9 - 舞蹈+夹爪
 10 - 夹爪测试         11 - 抓取放置         12 - RGB灯光
 13 - 多关节联动       14 - 画正方形         15 - 呼吸效果
 16 - 开始录制         17 - 停止录制         18 - 回放轨迹
 19 - 保存轨迹         20 - 加载轨迹
  0 - 返回上级
"""
    while True:
        print(menu)
        c = input("请选择: ").strip()
        if c == "0": break
        try:
            if c == "1": back_zero()
            elif c == "2": relax_arms()
            elif c == "3":
                j = int(input("关节编号(1-6): ")); a = float(input("角度: "))
                single_joint_move(j, a)
            elif c == "4":
                x = float(input("X: ") or 150); y = float(input("Y: ") or -130)
                move_to_coords(x, y)
            elif c == "5": head_shake()
            elif c == "6": head_nod()
            elif c == "7": head_dance()
            elif c == "8": dance_performance()
            elif c == "9": dance_with_gripper()
            elif c == "10": gripper_test()
            elif c == "11": pick_and_place_demo()
            elif c == "12": rgb_cycle()
            elif c == "13": multi_joint_move()
            elif c == "14": draw_square()
            elif c == "15": breathing()
            elif c == "16": start_recording()
            elif c == "17": stop_recording()
            elif c == "18": play_trajectory()
            elif c == "19": save_trajectory()
            elif c == "20": load_trajectory()
        except Exception as e:
            print(f"[ERROR] 动作执行失败: {e}")


def main():
    global mc

    print("=" * 50)
    print("  机械臂综合控制平台")
    print("=" * 50)
    print("  1 — 真实机械臂（Socket 连接）")
    print("  2 — 仿真模式")
    print("  3 — MuJoCo 3D 物理仿真")
    c = input("\n请选择: ").strip()

    if c == "1":
        ip = input(f"IP [{ROBOT_IP}]: ").strip() or ROBOT_IP
        port = input(f"端口 [{ROBOT_PORT}]: ").strip() or ROBOT_PORT
        if not connect_robot(ip, int(port)):
            print("连接失败，切换为仿真模式")
            mc = SimRobot()
            interruptible_sleep(0.5)
    elif c == "2":
        mc = SimRobot()
        interruptible_sleep(0.5)
    elif c == "3":
        try:
            mc = Mujoco3DRobot()
        except Exception as e:
            print(f"[ERROR] 启动 3D 仿真失败: {e}")
            print("切换为普通仿真模式")
            mc = SimRobot()
        interruptible_sleep(0.5)
    else:
        print("无效选择")
        return

    action_menu()

    if isinstance(mc, Mujoco3DRobot):
        input("\n按 Enter 关闭 3D 窗口...")
        mc.close()
    elif isinstance(mc, SimRobot):
        print("[SIM] 仿真结束")
    else:
        mc.set_color(0, 0, 0)
    print("程序退出")


if __name__ == "__main__":
    main()
