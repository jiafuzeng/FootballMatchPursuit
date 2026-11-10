# FootballMatchPursuit

把一场足球转播写成 [事件说明书](事件说明书.md) 里的事件 JSON。字段样子见 [test_data/test.json](test_data/test.json)。

做法是五步：先标出谁在画面上，再找出动作发生的时刻，接着截出片段，最后让云端模型读画面上已经画好的号码和坐标，并给传球补上接球人。模型不自己算位置。

默认这场是 `test_data/1-2-6.mp4`。环境是 conda `osl`。Qwen 的地址和密钥放在项目根目录 `.env`，复制 `.env.example` 后填写，不写入仓库。视频用 Git LFS。

## 五步

| 步 | 脚本 | 做什么 | 写出 |
|---|---|---|---|
| 0 | `step0_轨迹坐标.py` | 在原画面上画脚底圆环、追踪号、头顶中心 `[x,y]`。足球和裁判不跟踪 | `outputs/opensportslib/1-2-6.step0.mp4` |
| 1 | `step1_事件监测.py` | 用原始视频定位动作时刻。看的是比赛，不看 step0 画的字 | `outputs/opensportslib/test.step1.json` |
| 2 | `step2_事件截帧.py` | 按定位时刻，从带标注的视频上每个事件截 21 帧 | `outputs/opensportslib/clips/1-2-6.step0/` |
| 3 | `step3_控球识别-qwen38-max.py` | Qwen3.8-Max 读画面：球衣号、追踪号、坐标、红白队、比分牌时钟 | `outputs/opensportslib/1-2-6.step0.step3.qwen38-max.json` |
| 4 | `step4_接球识别-qwen38-max.py` | 只给传球和长传补接球人。其他字段从 step3 原样抄来 | `outputs/opensportslib/1-2-6.step0.step4.qwen38-max.json` |

step1 的 `frame` 是每秒 2 帧的采样序号。从 step3 起，JSON 里的 `frame` 是原视频帧号。`name_cn` 由队名和动作拼成，中间没有空格，例如 `Team A长传`。看不清的号码写 `unknown`。

step4 只问传球和长传。出球前留 15 帧；传球再看 3 秒，长传再看 5 秒。按每秒 5 帧抽帧，再慢放成 3 秒或 5 秒。片段在 `outputs/opensportslib/clips/step4/`。头球、停球（带球）等不截片段，接球人写 `unknown`。模型读到的追踪号若和传球人相同，也写成 `unknown`。

一条控球记录：

```json
{
  "frame": 300,
  "player_code": "unknown",
  "player_id": "7",
  "name_cn": "Team A长传",
  "player_screen_xy": [479.0, 482.0],
  "action_screen_time_period": "15:36"
}
```

step4 在 `name_cn` 后面加上 `receiver_code` 和 `receiver_id`。

## 怎么跑

```bash
conda activate osl
python step0_轨迹坐标.py
python step1_事件监测.py
python step2_事件截帧.py
python step3_控球识别-qwen38-max.py
python step4_接球识别-qwen38-max.py
```

后一步读前一步的默认结果。step0、step1 可用 `--max-seconds` 只跑开头。step3、step4 可用 `--index` 只跑一条。

球衣号、追踪号、坐标和时钟都是模型看画面读出来的，还没有人工复核。说明书要求下一脚触球人的追踪号等于这次传球的接球人，这条对应关系现在没有核对。

跨片段的核对还没做。以后如果要看一整段进攻怎么发展，再评估 [VLX-VR](VLX-VR/)。
