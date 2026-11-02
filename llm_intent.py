# -*- encoding:utf-8 -*-
import json
import dashscope
from dashscope import Generation

SYSTEM_PROMPT = """你是一个机械臂控制助手。根据用户的自然语言指令，输出JSON格式的动作命令。

可用动作:
  - back_zero: 机械臂归零，回到初始位置
  - relax_arms: 放松所有关节，释放舵机
  - stop: 急停，立即停止所有动作并释放舵机
  - head_shake: 左右摆头/摇头
  - head_nod: 点头
  - head_dance: 头部舞蹈
  - dance: 舞蹈表演，多个姿态交替
  - dance_gripper: 舞蹈配合夹爪开合
  - pick_place: 抓取放置演示
  - rgb_light: RGB灯光循环
  - light_off: 关灯，熄灭灯光
  - gripper_test: 夹爪开合测试
  - draw_square: 画正方形轨迹
  - breathing: 呼吸灯效果
  - record_start: 开始录制机械臂轨迹
  - record_stop: 停止录制
  - replay: 回放已录制的轨迹
  - save: 保存轨迹到文件
  - load: 从文件加载轨迹

输出格式必须是纯JSON，不要加任何其他文字:
{"action": "动作名", "reply": "简短口语回复"}

如果用户说的话不匹配任何动作，action 填 "unknown"，reply 简短提示支持的操作。"""


class IntentParser:
    """阿里云 qwen 意图解析器：自然语言 → 结构化动作"""

    def __init__(self, api_key=None, model="qwen-turbo"):
        if api_key:
            dashscope.api_key = api_key
        self.model = model

    def parse(self, text):
        """将自然语言文本解析为机械臂动作，返回 (action, reply)"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]

        result = ""
        try:
            response = Generation.call(
                model=self.model,
                messages=messages,
                result_format="message",
                temperature=0.1,
                max_tokens=256,
            )

            if response.status_code != 200:
                return "unknown", f"调用失败(HTTP {response.status_code})"

            result = response.output.choices[0].message.content.strip()
            # 容错：去掉 markdown 代码块标记
            if result.startswith("```"):
                result = result.split("\n", 1)[1]
                if result.endswith("```"):
                    result = result[:-3]
                result = result.strip()

            data = json.loads(result)
            return data.get("action", "unknown"), data.get("reply", "")
        except json.JSONDecodeError:
            return "unknown", result[:100] if result else "未理解"
        except Exception as e:
            return "unknown", str(e)[:50]
