"""定位模型类别到中文动作。顺序与 OSL-loc-snbas-2025-e2e 的 config.yaml 一致。"""

LABEL_TO_ACTION = {
    "PASS": "传球",
    "DRIVE": "停球(带球)",
    "HEADER": "头球",  # 可能是传球或射门
    "HIGH PASS": "长传",
    "OUT": "出界",  # 界外手抛球是 THROW IN
    "CROSS": "传中",  # 与 PASS、HIGH PASS 并列
    "THROW IN": "界外手抛球",
    "SHOT": "射门",
    "BALL PLAYER BLOCK": "封堵",  # 抢断是 PLAYER SUCCESSFUL TACKLE
    "PLAYER SUCCESSFUL TACKLE": "抢断",
    "FREE KICK": "任意球",  # 角球只是其中一种
    "GOAL": "进球",
}
