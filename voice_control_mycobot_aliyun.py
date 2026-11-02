# -*- encoding:utf-8 -*-
# 语音控制机械臂 — 阿里云实时语音识别 + LLM 意图解析 + 三种后端（真机/仿真/3D）
import sys
import os
import base64
import time
import threading
import logging
import msvcrt
import sounddevice as sd
import numpy as np
import dashscope
from dashscope.audio.asr import *

from llm_intent import IntentParser
import mycobot_controller as mc

# ---- 阿里云百炼配置（密钥从环境变量读取，不写入源码）----
WORKSPACE_ID = os.environ.get("DASHSCOPE_WORKSPACE_ID", "")
dashscope.base_websocket_api_url = (
    f"wss://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK = 3200  # 100ms


class AsrCallback(RecognitionCallback):
    """ASR 回调：识别结果写入外部队列"""

    def __init__(self, task_queue):
        self.task_queue = task_queue
        self.last_partial = ""

    def on_open(self):
        print("[ASR] 连接成功")

    def on_close(self):
        print("[ASR] 连接关闭")

    def on_complete(self):
        pass

    def on_error(self, result):
        print(f"[ASR错误] {result.message}")

    def on_event(self, result):
        sentence = result.get_sentence()
        if 'text' not in sentence:
            return
        text = sentence['text'].strip()
        if RecognitionResult.is_sentence_end(sentence):
            self.last_partial = ""
            if text and text not in ("。", "，", "？", "！", ".", ",", "?", "!"):
                print(f"  → {text}")
                self.task_queue.append(("text", text))
        else:
            self.last_partial = text


class VoiceRobotClient:
    """阿里云语音控制机械臂 — 支持真机 / 仿真 / MuJoCo 3D"""

    def __init__(self, robot_mode="sim"):
        self.robot_mode = robot_mode
        self.task_queue = []
        self.action_thread = None
        self.running = True
        self.robot = None
        self.intent_parser = IntentParser()
        self.is_executing = False
        self.recognition = None
        self.mic_stream = None

    def connect_robot(self):
        """根据模式初始化底层机械臂后端"""
        if self.robot_mode == "real":
            while True:
                print(f"\n[机械臂] 正在连接 {mc.ROBOT_IP}:{mc.ROBOT_PORT} ...")
                if mc.connect_robot():
                    self.robot = mc.mc
                    return True

                # 连接失败：让用户选择下一步，而不是静默回退
                print("\n" + "-" * 50)
                print("  [连接失败] 无法连接真实机械臂")
                print("  请检查: 机械臂是否上电 / IP 是否正确 / 是否同一局域网")
                print("  请选择:")
                print("    1 — 重试连接")
                print("    2 — 切换到仿真模式")
                print("    3 — 切换到 MuJoCo 3D 仿真")
                print("    4 — 退出程序")
                c = input("  请输入 (1/2/3/4): ").strip()
                if c == "1":
                    continue
                elif c == "2":
                    print("[机械臂] 已切换到仿真模式")
                    self.robot_mode = "sim"
                    break
                elif c == "3":
                    print("[机械臂] 已切换到 MuJoCo 3D 仿真")
                    self.robot_mode = "3d"
                    break
                else:
                    print("[机械臂] 已取消连接")
                    return False
        if self.robot_mode == "3d":
            try:
                self.robot = mc.Mujoco3DRobot()
                mc.mc = self.robot
                return True
            except Exception as e:
                print(f"[机械臂] 启动 3D 仿真失败: {e}")
                print("[机械臂] 回退普通仿真")
                self.robot_mode = "sim"
        if self.robot_mode == "sim":
            self.robot = mc.SimRobot()
            mc.mc = self.robot
            return True
        return False

    def _execute_action(self, action_name, reply=""):
        try:
            print(f"\n{'='*30}")
            print(f"[执行] {action_name}")
            if reply:
                print(f"[回复] {reply}")

            if action_name == "unknown":
                print("[未识别] 不知道执行什么动作")
                return

            self.is_executing = True
            if action_name == "back_zero":      mc.back_zero()
            elif action_name == "relax_arms":   mc.relax_arms()
            elif action_name == "stop":         mc.emergency_stop()
            elif action_name == "head_nod":     mc.head_nod()
            elif action_name == "head_shake":   mc.head_shake()
            elif action_name == "head_dance":   mc.head_dance()
            elif action_name == "dance":        mc.dance_performance()
            elif action_name == "dance_gripper": mc.dance_with_gripper()
            elif action_name == "pick_place":   mc.pick_and_place_demo()
            elif action_name == "rgb_light":    mc.rgb_cycle()
            elif action_name == "gripper_test": mc.gripper_test()
            elif action_name == "draw_square":  mc.draw_square()
            elif action_name == "breathing":    mc.breathing()
            elif action_name == "record_start": mc.start_recording()
            elif action_name == "record_stop":  mc.stop_recording()
            elif action_name == "replay":       mc.play_trajectory()
            elif action_name == "save":         mc.save_trajectory()
            elif action_name == "load":         mc.load_trajectory()
            self.is_executing = False
            print(f"[完成]")
            print(f"{'='*30}\n")
        except mc.ActionInterrupted:
            self.is_executing = False
            print(f"[打断] {action_name} 被新指令打断")
            print(f"{'='*30}\n")
        except Exception as e:
            self.is_executing = False
            print(f"[动作失败] {action_name}: {e}")

    def _parse_text(self, text):
        print(f"[意图] 正在理解: 「{text}」", flush=True)
        try:
            action, reply = self.intent_parser.parse(text)
        except Exception:
            action, reply = "unknown", ""
        if action == "unknown":
            action, reply = self._fallback_match(text)
        print(f"[意图] → {action} | {reply}", flush=True)
        return action, reply

    def _fallback_match(self, text):
        mapping = [
            (["归零", "回零", "原点", "初始", "回到"], "back_zero"),
            (["放松", "释放", "松开"], "relax_arms"),
            (["急停", "紧急停止", "紧急停", "刹车", "马上停", "立刻停"], "stop"),
            (["点头"], "head_nod"),
            (["摆头", "摇头", "晃头"], "head_shake"),
            (["头部舞蹈", "头部跳舞"], "head_dance"),
            (["跳舞", "舞蹈", "跳个舞", "来段舞"], "dance"),
            (["夹爪舞", "爪子舞", "跳舞夹爪"], "dance_gripper"),
            (["抓取", "放置", "搬运", "抓一下", "夹一下", "拿东西"], "pick_place"),
            (["灯光", "彩灯", "亮灯", "变色", "开灯"], "rgb_light"),
            (["夹爪", "爪子", "夹子", "开合"], "gripper_test"),
            (["正方形", "画方", "画个方"], "draw_square"),
            (["呼吸", "呼吸灯"], "breathing"),
            (["录制", "开始录"], "record_start"),
            (["停止录制", "结束录制"], "record_stop"),
            (["回放", "播放轨迹", "重放"], "replay"),
            (["保存", "存"], "save"),
            (["加载", "载入"], "load"),
        ]
        for keywords, action in mapping:
            for kw in keywords:
                if kw in text:
                    return action, f"已理解: {kw}"
        return "unknown", f"未识别「{text}」"

    def _action_worker(self):
        print("[动作线程] 已启动")
        self._action_thread = None
        while self.running:
            try:
                if self.task_queue:
                    # 只保留最新一条指令，清空积压
                    task = self.task_queue.pop(0)
                    while self.task_queue:
                        self.task_queue.pop(0)

                    if task[0] != "text":
                        continue

                    action, reply = self._parse_text(task[1])

                    if action == "unknown":
                        self._execute_action(action, reply)
                        continue

                    # 打断上一个动作
                    mc.request_interrupt()
                    if self._action_thread and self._action_thread.is_alive():
                        self._action_thread.join(timeout=3)

                    # 启动新动作
                    mc.clear_interrupt()
                    self._action_thread = threading.Thread(
                        target=self._execute_action,
                        args=(action, reply),
                        daemon=True,
                    )
                    self._action_thread.start()
            except Exception as e:
                print(f"[动作线程异常] {e}")
            time.sleep(0.05)
        print("[动作线程] 已停止")

    def start(self):
        if not self.connect_robot():
            print("[错误] 机械臂初始化失败")
            return

        self.action_thread = threading.Thread(target=self._action_worker, daemon=True)
        self.action_thread.start()

        print(f"[设备] {sd.query_devices(kind='input')['name']}")
        print(f"[麦克风] 采样率:{SAMPLE_RATE} 位深:16bit 声道:单声道\n")

        while self.running:
            while self.is_executing and self.running:
                time.sleep(0.1)
            if not self.running:
                break

            self._record_one()

    def _record_one(self):
        print("[录音] 请说话... 按 Enter 停止，Ctrl+C 退出")

        callback = AsrCallback(self.task_queue)
        self.recognition = Recognition(
            model='paraformer-realtime-v2',
            format='pcm',
            sample_rate=SAMPLE_RATE,
            language_hints=['zh', 'en'],
            callback=callback,
        )
        self.recognition.start()

        try:
            self.mic_stream = sd.InputStream(
                samplerate=SAMPLE_RATE, channels=CHANNELS,
                dtype='int16', blocksize=CHUNK
            )
            self.mic_stream.start()
        except Exception as e:
            print(f"[麦克风错误] {e}")
            self.recognition.stop()
            return

        try:
            while self.running:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key in (b'\r', b'\n'):
                        print("\n[停止录音]")
                        break

                data, _ = self.mic_stream.read(CHUNK)
                self.recognition.send_audio_frame(data.tobytes())

        except Exception as e:
            print(f"\n[录音异常] {e}")
        finally:
            self.mic_stream.stop()
            self.mic_stream.close()
            self.mic_stream = None
            self.recognition.stop()

        # 补偿：句末标点可能没触发 is_sentence_end
        leftover = callback.last_partial.strip()
        if leftover and leftover not in ("。", "，", "？", "！", ".", ",", "?", "!"):
            print(f"[补偿] 「{leftover}」")
            self.task_queue.append(("text", leftover))

        # 等待处理完毕
        while self.task_queue and self.running:
            time.sleep(0.1)

        time.sleep(0.5)

    def shutdown(self):
        self.running = False
        if self.mic_stream:
            try:
                self.mic_stream.stop()
                self.mic_stream.close()
            except Exception:
                pass
            self.mic_stream = None
        try:
            if isinstance(self.robot, mc.Mujoco3DRobot):
                print("[3D] 关闭 MuJoCo 窗口")
                self.robot.close()
            elif isinstance(self.robot, mc.SimRobot):
                print("[SIM] 仿真结束")
            else:
                self.robot.set_color(0, 0, 0)
        except Exception:
            pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("websocket").setLevel(logging.WARNING)

    print("=" * 50)
    print("  语音控制机械臂系统 (阿里云Paraformer)")
    print("=" * 50)
    print("  1 — 仿真模式")
    print("  2 — 真实机械臂")
    print("  3 — MuJoCo 3D 物理仿真")
    choice = input("\n请选择: ").strip()
    if choice == "2":
        robot_mode = "real"
    elif choice == "3":
        robot_mode = "3d"
    else:
        robot_mode = "sim"

    client = VoiceRobotClient(robot_mode)
    try:
        client.start()
    except KeyboardInterrupt:
        pass
    finally:
        client.running = False
        time.sleep(0.3)
        client.shutdown()
        print("[程序] 已退出")
