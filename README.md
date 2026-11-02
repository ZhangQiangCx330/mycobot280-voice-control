# myCobot 280 Web 集成控制平台（实训项目）

基于阿里云语音识别（Paraformer）+ 通义千问（qwen）大模型的 myCobot 280 机械臂控制平台。
后端用 MuJoCo 计算正运动学位姿，支持 **浏览器 Web 控制**、**键盘控制台**、**命令行语音控制** 三种使用方式。

## 启动

### 方式 1：Web 控制（浏览器）

```bash
python web_server.py
```

浏览器打开：

```
http://localhost:5000
```

页面提供 Three.js 3D 机械臂、动作按钮、语音（按住说话，松开识别）和日志区。

### 方式 2：键盘控制台

```bash
python mycobot_controller.py
```

启动后选择运行模式：

```
  1 — 真实机械臂（Socket 连接）
  2 — 仿真模式
  3 — MuJoCo 3D 物理仿真
```

### 方式 3：命令行语音控制

```bash
python voice_control_mycobot_aliyun.py
```

启动后选择运行模式：

```
  1 — 仿真模式
  2 — 真实机械臂
  3 — MuJoCo 3D 物理仿真
```

选 2（真机）连接失败时，会提示选择重试 / 切仿真 / 切 3D / 退出。按住说话、松开识别即可。


## 功能特性

- **3D 可视化**：浏览器内 Three.js 3D 机械臂（官方 STL 模型，可旋转 / 缩放视角）；命令行下可用 MuJoCo 3D 物理仿真
- **语音控制**：按住说话 → 阿里云 ASR 识别 → qwen 意图解析 → 执行动作
- **动作打断**：执行中发新指令会立即打断
- **动作按钮**：18 个动作 + 急停 + 关灯
- **RGB 灯指示器**：控制区实时显示颜色
- **MuJoCo 正运动学**：无头计算各关节位姿实时推送前端

## 架构

```
浏览器 (Three.js 3D + 麦克风)
   ├─ /ws         控制通道：动作指令 ↔ 位姿 / 日志 / 颜色推送
   └─ /ws_audio   音频通道：录音 PCM → 阿里云 ASR
后端 Flask + Flask-Sock
   ├─ MuJoCo 正运动学（编译后 STL 顶点 + geom 位姿 → Three.js 坐标）
   └─ 阿里云 ASR + qwen 意图 + 动作执行

命令行入口直接复用同一套动作库（mycobot_controller.py），
   ├─ 键盘控制：mycobot_controller.py（1真机 / 2仿真 / 3MuJoCo）
   └─ 语音控制：voice_control_mycobot_aliyun.py（按住说话）
```

## 目录结构

```
code/
├── web_server.py                    # Flask 后端（Web 入口）
├── voice_control_mycobot_aliyun.py  # 命令行语音控制入口
├── mycobot_controller.py            # 动作库（键盘控制台入口，三种机器人后端）
├── llm_intent.py                    # 阿里云 qwen 意图解析
├── mycobot_mujoco.py                # MuJoCo 3D 引擎封装（命令行 3D 用）
├── templates/index.html             # 前端页面（Three.js）
├── static/js/                       # 本地 Three.js / OrbitControls / STLLoader
├── static/models/                   # 机械臂模型（MJCF + STL，Web 用）
├── xml/                             # MJCF 模型（命令行 MuJoCo 用）
└── meshes_mujoco/                   # 各关节 STL 网格（命令行 MuJoCo 用）
```


## 依赖安装

```bash
pip install flask flask-sock dashscope mujoco numpy sounddevice
```

## 配置

### 阿里云 API Key / Workspace ID

语音功能需要阿里云百炼的密钥，从环境变量读取（源码不含密钥）：

```bash
set DASHSCOPE_API_KEY=你自己的密钥
set DASHSCOPE_WORKSPACE_ID=ws-你的工作空间ID
```

Windows 下也可通过「系统属性 → 高级 → 环境变量」设置，重启终端后生效。

### 机械臂 IP

编辑 `mycobot_controller.py` 顶部的：

```python
ROBOT_IP = "192.168.x.x"   # 改成你的机械臂 IP
ROBOT_PORT = 9000
```

## 使用方式

- Web 页：鼠标拖动旋转视角，滚轮缩放 3D 机械臂；点击按钮执行动作；「按住说话」录音（首次需允许麦克风权限）
- 命令行语音：按住任意键说话、松开识别，用自然语言控制

## 动作列表

| 按钮 / 语音 | 说明 |
|------|------|
| 归零 / 放松 / 急停 | 基础控制 |
| 点头 / 摇头 / 头部舞蹈 | 头部动作 |
| 跳舞 / 跳舞+夹爪 / 呼吸灯 | 表演动作 |
| 抓取放置 / 夹爪测试 | 夹爪动作 |
| RGB灯光 / 关灯 / 画正方形 | 灯光与创意 |
| 开始录制 / 停止录制 / 回放 / 保存 / 加载 | 轨迹示教 |

## 注意事项

- 项目目录路径**不要包含中文**（MuJoCo 无法读取含中文的路径）。
- 选「真实机械臂」需机械臂已上电并连到同一网络；连不上会自动降级到仿真。
- 3D 机械臂为 MuJoCo 编译后的模型数据，前端用简化位姿摆放，物理渲染质量等同于 MuJoCo。
- 语音识别结果通过日志区显示，机械臂动作实时在 3D 中展示。