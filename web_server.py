# -*- encoding:utf-8 -*-
"""
myCobot 280 Web 集成后端
Flask + Flask-Sock + MuJoCo 正运动学 + 阿里云 ASR

功能:
  - 浏览器 Three.js 3D 显示机械臂（后端用 MuJoCo 计算位姿推送给前端）
  - WebSocket 控制通道: 动作指令 + 位姿/日志推送
  - WebSocket 音频通道: 浏览器录音 -> 阿里云 ASR -> qwen 意图 -> 动作

启动: python web_server.py
访问: http://localhost:5000
"""

import os
import json
import math
import struct
import base64
import threading
import time

import numpy as np
import mujoco
import dashscope
from flask import Flask, render_template
from flask_sock import Sock
from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

import mycobot_controller as mc
from llm_intent import IntentParser

# ==================== 阿里云配置 ====================
# 密钥从环境变量读取，不写入源码

WORKSPACE_ID = os.environ.get("DASHSCOPE_WORKSPACE_ID", "")
dashscope.base_websocket_api_url = (
    f"wss://{WORKSPACE_ID}.cn-beijing.maas.aliyuncs.com/api-ws/v1/inference"
)
dashscope.api_key = os.environ.get("DASHSCOPE_API_KEY", "")

SAMPLE_RATE = 16000
CHANNELS = 1

# ==================== Flask 应用 ====================

app = Flask(__name__, static_folder="static", static_url_path="/static")
sock = Sock(app)

# ==================== MuJoCo 正运动学核心 ====================

MODEL_XML = os.path.join(os.path.dirname(__file__), "static", "models", "mycobot_280jn_mujoco.xml")

_mj_model = mujoco.MjModel.from_xml_path(MODEL_XML)
_mj_data = mujoco.MjData(_mj_model)
mujoco.mj_forward(_mj_model, _mj_data)

# 找到每个 mesh 对应的 visual geom（用于提取世界位姿）
_mesh_geoms = {}
for gid in range(_mj_model.ngeom):
    if _mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
        if _mj_model.geom_contype[gid] == 0:  # visual geom
            mid = _mj_model.geom_dataid[gid]
            name = mujoco.mj_id2name(_mj_model, mujoco.mjtObj.mjOBJ_MESH, mid)
            if name and name not in _mesh_geoms:
                _mesh_geoms[name] = gid


# MuJoCo 坐标系 (z 向上) 转 Three.js 坐标系 (y 向上) 的旋转矩阵：绕 x 轴 -90°
_R_COORD = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])


def _compute_links():
    """
    计算 7 个 mesh 的世界位姿，转换为 Three.js 坐标系（y 向上）。
    前端加载的是 MuJoCo 编译后的顶点（已居中+对齐），
    因此这里只需 geom 的世界位姿，无需再处理 refpos/refquat。
    """
    links = []
    for name, gid in _mesh_geoms.items():
        p_g = _mj_data.geom_xpos[gid]
        R_g = _mj_data.geom_xmat[gid].reshape(3, 3)

        pos = _R_COORD @ p_g  # (x, y, z) -> (x, z, -y)
        R_three = _R_COORD @ R_g  # 主动旋转，直接左乘

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R_three).reshape(9))

        links.append({
            "mesh": name,
            "pos": [round(float(v), 6) for v in pos],
            "quat": [round(float(v), 6) for v in quat.tolist()],  # w,x,y,z
        })
    return links


def _generate_stl_bytes(mesh_name):
    """用 MuJoCo 编译后的顶点和面生成 binary STL"""
    mid = mujoco.mj_name2id(_mj_model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    vadr = _mj_model.mesh_vertadr[mid]
    verts = _mj_model.mesh_vert[vadr:vadr + _mj_model.mesh_vertnum[mid]]
    fadr = _mj_model.mesh_faceadr[mid]
    faces = _mj_model.mesh_face[fadr:fadr + _mj_model.mesh_facenum[mid]]

    out = bytearray()
    out += b"\x00" * 80
    out += struct.pack("<I", len(faces))
    for tri in faces:
        v0 = verts[tri[0]]
        v1 = verts[tri[1]]
        v2 = verts[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(n)
        n = n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])
        out += struct.pack("<3f", n[0], n[1], n[2])
        for v in (v0, v1, v2):
            out += struct.pack("<3f", v[0], v[1], v[2])
        out += struct.pack("<H", 0)
    return bytes(out)


def _set_mj_angles(angles_deg):
    """设置 6 个关节角度（度），更新正运动学"""
    for i in range(6):
        _mj_data.qpos[i] = math.radians(angles_deg[i])
    mujoco.mj_forward(_mj_model, _mj_data)


# ==================== Web 客户端广播 ====================

_ws_clients = set()
_clients_lock = threading.Lock()


def broadcast(message: dict):
    """向所有控制通道客户端广播消息"""
    data = json.dumps(message, ensure_ascii=False)
    with _clients_lock:
        clients = list(_ws_clients)
    for ws in clients:
        try:
            ws.send(data)
        except Exception:
            pass


def broadcast_pose():
    broadcast({"type": "pose", "links": _compute_links()})


def broadcast_log(msg):
    broadcast({"type": "log", "message": msg})


# ==================== WebRobot 仿真后端（适配 pymycobot API） ====================

class WebRobot:
    """Web 仿真机械臂：内部用 MuJoCo 计算位姿并推送前端"""

    def __init__(self):
        self._angles = [0, 0, 0, 0, 0, 0]
        self._gripper = 0
        self._color = (0, 0, 0)

    def get_angles(self):
        return list(self._angles)

    def send_angles(self, angles, speed):
        self._angles = list(angles)
        _set_mj_angles(angles)
        broadcast_pose()
        print(f"[动作] 角度 -> {angles} 速度 {speed}")

    def send_angle(self, joint_id, angle, speed):
        if 1 <= joint_id <= 6:
            self._angles[joint_id - 1] = angle
            _set_mj_angles(self._angles)
            broadcast_pose()
        print(f"[动作] 关节{joint_id} -> {angle}°")

    def send_coords(self, coords, speed, mode=0):
        print(f"[动作] 坐标 -> {coords}（逆运动学暂未实现）")

    def get_coords(self):
        return [160, 0, 160, 0, 0, 0]

    def set_color(self, r, g, b):
        self._color = (r, g, b)
        broadcast({"type": "color", "r": r, "g": g, "b": b})

    def set_gripper_value(self, value, speed):
        self._gripper = max(0, min(100, value))

    def set_gripper_state(self, state, speed):
        pass

    def get_gripper_value(self):
        return self._gripper

    def is_gripper_moving(self):
        return False

    def is_in_position(self, angles, error=0):
        return True

    def release_all_servos(self):
        pass

    def set_encoder(self, joint_id, value, speed):
        pass

    def set_encoders(self, values, speed):
        pass

    def get_encoder(self, joint_id):
        return 2048

    def is_controller_connected(self):
        return 1


mc.mc = WebRobot()


# ==================== 动作分派 ====================

ACTION_MAP = {
    "back_zero": mc.back_zero,
    "relax_arms": mc.relax_arms,
    "stop": mc.emergency_stop,
    "head_nod": mc.head_nod,
    "head_shake": mc.head_shake,
    "head_dance": mc.head_dance,
    "dance": mc.dance_performance,
    "dance_gripper": mc.dance_with_gripper,
    "pick_place": mc.pick_and_place_demo,
    "rgb_light": mc.rgb_cycle,
    "light_off": mc.light_off,
    "gripper_test": mc.gripper_test,
    "draw_square": mc.draw_square,
    "breathing": mc.breathing,
    "record_start": mc.start_recording,
    "record_stop": mc.stop_recording,
    "replay": mc.play_trajectory,
    "save": mc.save_trajectory,
    "load": mc.load_trajectory,
}

_action_thread = None
_action_lock = threading.Lock()


def run_action(action, params=None):
    """在独立线程执行动作，支持打断上一个动作"""
    global _action_thread
    params = params or {}

    func = ACTION_MAP.get(action)
    if func is None:
        broadcast_log(f"[未知动作] {action}")
        return

    # 打断上一个动作
    mc.request_interrupt()
    with _action_lock:
        prev = _action_thread
    if prev and prev.is_alive():
        prev.join(timeout=3)

    mc.clear_interrupt()

    def _run():
        try:
            func(**params)
            broadcast_log("[完成]")
        except mc.ActionInterrupted:
            broadcast_log("[打断] 动作被新指令打断")
        except Exception as e:
            broadcast_log(f"[错误] {e}")

    with _action_lock:
        _action_thread = threading.Thread(target=_run, daemon=True)
        _action_thread.start()


# ==================== ASR 与意图解析 ====================

intent_parser = IntentParser()
_task_queue = []
_recognition = None
_recognition_lock = threading.Lock()


class AsrCallback(RecognitionCallback):
    def __init__(self):
        self.last_partial = ""

    def on_open(self):
        broadcast_log("[ASR] 已连接")

    def on_close(self):
        pass

    def on_complete(self):
        pass

    def on_error(self, result):
        broadcast_log(f"[ASR错误] {result.message}")

    def on_event(self, result):
        sentence = result.get_sentence()
        if "text" not in sentence:
            return
        text = sentence["text"].strip()
        if RecognitionResult.is_sentence_end(sentence):
            self.last_partial = ""
            if text and text not in ("。", "，", "？", "！", ".", ",", "?", "!"):
                broadcast_log(f"[识别] {text}")
                _task_queue.append(text)
        else:
            self.last_partial = text


def _voice_worker():
    """处理识别文本 -> 意图 -> 动作"""
    while True:
        try:
            if _task_queue:
                text = _task_queue.pop(0)
                action, reply = intent_parser.parse(text)
                if action == "unknown":
                    action, reply = _fallback_match(text)
                broadcast_log(f"[意图] {action} | {reply}")
                if action not in ("unknown",):
                    run_action(action)
        except Exception as e:
            broadcast_log(f"[语音线程异常] {e}")
        time.sleep(0.05)


def _fallback_match(text):
    mapping = [
        (["急停", "紧急停止", "刹车", "马上停", "立刻停"], "stop"),
        (["归零", "回零", "原点", "初始"], "back_zero"),
        (["放松", "释放", "松开"], "relax_arms"),
        (["点头"], "head_nod"),
        (["摆头", "摇头", "晃头"], "head_shake"),
        (["头部舞蹈"], "head_dance"),
        (["跳舞", "舞蹈"], "dance"),
        (["夹爪舞"], "dance_gripper"),
        (["抓取", "放置", "搬运", "拿东西"], "pick_place"),
        (["灯光", "彩灯", "变色", "开灯"], "rgb_light"),
        (["关灯", "灭灯", "熄灯"], "light_off"),
        (["夹爪", "爪子", "开合"], "gripper_test"),
        (["正方形", "画方"], "draw_square"),
        (["呼吸"], "breathing"),
        (["开始录", "录制"], "record_start"),
        (["停止录制"], "record_stop"),
        (["回放", "重放"], "replay"),
        (["保存", "存"], "save"),
        (["加载", "载入"], "load"),
    ]
    for keywords, action in mapping:
        for kw in keywords:
            if kw in text:
                return action, f"已理解: {kw}"
    return "unknown", f"未识别「{text}」"


# ==================== HTTP 路由 ====================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/processed/<name>.stl")
def processed_stl(name):
    """返回 MuJoCo 编译后（已居中+对齐）的 STL"""
    if name not in _mesh_geoms:
        return "", 404
    return _generate_stl_bytes(name), 200, {"Content-Type": "application/octet-stream"}


# ==================== WebSocket 控制通道 ====================

@sock.route("/ws")
def ws_control(ws):
    with _clients_lock:
        _ws_clients.add(ws)
    try:
        # 初始推送当前位姿
        ws.send(json.dumps({"type": "pose", "links": _compute_links()}, ensure_ascii=False))
        while True:
            data = ws.receive()
            if data is None:
                break
            cmd = json.loads(data)
            action = cmd.get("action", "")
            params = cmd.get("params", {})
            if action == "get_pose":
                ws.send(json.dumps({"type": "pose", "links": _compute_links()}, ensure_ascii=False))
            else:
                broadcast_log(f"[指令] {action} {params}")
                run_action(action, params)
    except Exception:
        pass
    finally:
        with _clients_lock:
            _ws_clients.discard(ws)


# ==================== WebSocket 音频通道 ====================

@sock.route("/ws_audio")
def ws_audio(ws):
    global _recognition
    with _recognition_lock:
        callback = AsrCallback()
        _recognition = Recognition(
            model="paraformer-realtime-v2",
            format="pcm",
            sample_rate=SAMPLE_RATE,
            language_hints=["zh", "en"],
            callback=callback,
        )
        _recognition.start()

    try:
        while True:
            data = ws.receive()
            if data is None:
                break
            msg = json.loads(data)
            mtype = msg.get("type")
            if mtype == "audio":
                pcm = base64.b64decode(msg.get("data", ""))
                if pcm:
                    _recognition.send_audio_frame(pcm)
            elif mtype == "stop":
                break
    except Exception:
        pass
    finally:
        with _recognition_lock:
            try:
                _recognition.stop()
                leftover = callback.last_partial.strip()
                if leftover and leftover not in ("。", "，", "？", "！", ".", ",", "?", "!"):
                    broadcast_log(f"[识别] {leftover}")
                    _task_queue.append(leftover)
            except Exception:
                pass
            _recognition = None


# ==================== 启动入口 ====================

if __name__ == "__main__":
    threading.Thread(target=_voice_worker, daemon=True).start()
    print("=" * 50)
    print("  myCobot 280 Web 集成平台")
    print("=" * 50)
    print("  浏览器打开: http://localhost:5000")
    print("  按 Ctrl+C 退出")
    print()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
