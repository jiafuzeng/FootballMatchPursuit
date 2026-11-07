# FootballMatchPursuit

把一段足球转播视频写成 [事件说明书](事件说明书.md) 规定的事件 JSON。当前流程是四步：先在画面上标出球员追踪号和中心坐标，再定位动作时间，然后从带标注的视频上截出每个事件，最后让视觉模型读画面，补上控球人。

字段样例是 [test_data/test.json](test_data/test.json)。更完整的系统边界见 [足球视频事件JSON系统概要设计.md](足球视频事件JSON系统概要设计.md)。

环境是 conda 环境 `osl`（Python 3.12，PyTorch 2.5.1+cu121）。这台机器是 GTX 1060 6GB，驱动只报告 CUDA 12.2，未安装 DALI。定位模型的 batch size 固定为 1。视频文件用 Git LFS 存放。

## 实现步骤

按这个顺序跑。后一步读取前一步的默认产物。

### step0 轨迹坐标

`step0_轨迹坐标.py` 读取原始比赛视频，默认 `test_data/test.mp4`。它调用 `sports` 的球员跟踪，在原画面的拷贝上只加三样东西：脚底圆环、圆环里的追踪号、头顶的人物中心点 `[x,y]`。坐标是人物框中心，x 向右、y 向下，取整。足球和裁判不跟踪，也不画圆环。原片像素不改，结果用 H.264、yuv420p 写出，索引放在文件头。

默认输出 `outputs/opensportslib/test.step0.mp4`。

### step1 事件监测

`step1_事件监测.py` 把原始视频交给 [OpenSportsLab/OSL-loc-snbas-2025-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e)。默认输入仍是 `test_data/test.mp4`，因为定位看的是比赛内容，不看 step0 画上的数字。模型按每秒 2 帧抽帧，每 100 帧一个窗口。保存的是非极大值抑制之后、置信度不低于 `--min-confidence`（默认 0.90）的动作点，类别按 `label_map.py` 写成中文：传球、停球(带球)、头球、长传、出界、传中、界外手抛球、射门、封堵、抢断、任意球、进球。

`frame` 是模型 2 帧/秒的采样序号，原样保留。`position_ms` 是动作时刻。默认输出 `outputs/opensportslib/test.step1.json`。这一步不识别人，也不写画面坐标。角球、扑救、助攻不在这 12 类里。

### step2 事件截帧

`step2_事件截帧.py` 读取 step1 的 JSON，默认 `outputs/opensportslib/test.step1.json`。截取用的视频单独配置，默认是已经画好追踪号和中心坐标的 `outputs/opensportslib/test.step0.mp4`。

每个事件以 `round(position_ms / 1000 × 帧率)` 为中心，前后各 10 帧，连同中心帧一共 21 帧，按原帧率写成一个 mp4。step1 的 `frame` 原样抄进清单，不换算。原视频帧号写在 `source_frame`，片段范围写在 `start_frame` 和 `end_frame`。

默认片段目录是 `outputs/opensportslib/clips/test.step0/`，清单是同目录的 `step2.json`。

### step3 控球识别

`step3_控球识别-qwen38-max.py` 读取 `outputs/opensportslib/clips/test.step0/step2.json` 和同目录片段。模型是云端 Qwen3.8-Max。事件帧静图用来读数字，片段用来判断谁在控球。接口不收短于 2 秒的视频，脚本只把原帧放慢到 2 秒以上，不改画面。

模型从画面读取：

- `player_code`：控球人球衣号，看不清写 `unknown`
- `player_id`：脚底圆环里的追踪号
- `player_screen_xy`：这个人头顶方括号里的两个数
- 衣色：红衣 Team A，白衣 Team B
- `action_screen_time_period`：比分牌下方的比赛时钟，按画面原样保留

`name_cn` 由衣色和 step2 的动作拼成，中间没有空格，例如 `Team A长传`。输出记录里的 `frame` 用 step2 的 `source_frame`，也就是原视频帧号。默认结果是 `outputs/opensportslib/test.step0.step3.qwen38-max.json`，一条记录六个字段：

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

接口地址和密钥放在项目根目录的 `.env`，不写入仓库。复制 `.env.example` 后填入：

```text
BASE_URL=https://api.agicto.cn/v1
API_KEY=
```

## 现在怎么跑

```bash
conda activate osl
python step0_轨迹坐标.py
python step1_事件监测.py
python step2_事件截帧.py
python step3_控球识别-qwen38-max.py
```

step0、step1 可以用 `--max-seconds` 只处理开头若干秒。step2 换标注视频时用 `--video`。step3 只跑一个事件时用 `--index`。

| 文件 | 内容 |
|---|---|
| `outputs/opensportslib/test.step0.mp4` | 带脚底追踪号和头顶中心坐标的视频 |
| `outputs/opensportslib/test.step1.json` | 动作定位。`frame` 是 2 帧/秒采样序号，`position_ms` 是动作时刻 |
| `outputs/opensportslib/clips/test.step0/` | 每个事件 21 帧的标注片段，以及 `step2.json` |
| `outputs/opensportslib/test.step0.step3.qwen38-max.json` | 控球记录。`frame` 是原视频帧号 |

## 当前结果还不能直接交付

step3 读到的球衣号、追踪号、坐标和时钟来自模型看画面，还没有人工复核。接球人的 `receiver_code`、`receiver_id` 还没写。说明书要求用同一个追踪号把传球和下一脚停球串起来，抢断、射门、助攻、扑救也有成对关系，这些关系现在都还没有。

## 后期调研：VLX-VR

当前 step3 只看每个事件自己的 21 帧，接球人和跨片段关系还串不起来。下一步调研 [VLX-VR](VLX-VR/)（Video Reasoning），用它做允许等待的整段视频语义分析。需要持续采样、快速反馈或实时告警时，评估同系列的 VLX-Flow。

### VLX-VR 是什么

VLX-VR 是面向复杂视频分析的端云协同深度推理模型。它围绕用户的问题定位关键时间段，把画面、时间线、文本和数值线索合在一起，做跨片段、多步推理，并返回可以回看的分析结果。

它适合已经落成文件、可以等一段时间、又需要完整语义的视频。一场约 10 分钟的转播，可以问「按时间顺序列出关键事件和时间段」「哪一方占据优势」「这脚球是怎么发展到射门的」。模型在回答里给出关键时间点或视频依据，方便和定位结果对照。

公开评测来自 MINERVA 复杂视频推理基准的一次对照：总体准确率 78.79%。按原始视频时长分成三档，5 分钟以下 76.70%，5 至 15 分钟 78.73%，15 分钟以上 80.92%。在模型答对的样本里，推理轨迹与人工参考轨迹及证据描述的一致性是 96.20%。这些数字只对应该次评测集、版本和统计口径。96.20% 是答对样本中的轨迹一致性，要在本项目的比赛样本上再验。

### 能做什么，边界在哪里

| 能力 | 对本项目的含义 |
|---|---|
| 关键时间段定位 | 按问题找出和结论相关的片段，可与 OpenSportsLib 的 `position_ms` 对照 |
| 跨片段事件梳理 | 结合前后状态，说明一次进攻如何发生、发展 |
| 多步视频推理 | 综合画面、时间和情境，写出自然语言结论 |
| 证据回看 | 回答里带时间点或视频依据，供人工复核 |

效果受视频时长、码率、清晰度、资源配置和问题复杂度影响，以平台当时的校验为准。生产使用仍要保留业务规则、检测或追踪、人工复核和本项目样本上的验收。VLX-VR 的输出是分析结论和时间证据；说明书里的 `player_id`、固定字段顺序和逐条 JSON，仍由本项目的适配和复核来完成。

### 怎么接

页面体验可以上传本地视频或使用平台预设视频，输入自然语言问题做快速验证。正式接入用 API/SDK，由业务侧处理等待、超时、重试和结果复核。

API 不上传文件，也不接受 `file_id`、本地路径或平台文件柜字段。调用方先把视频放到对象存储或文件服务，得到模型服务能直接读取的 URL，再传给 `video_url.url`。地址在请求处理期间保持可访问，使用平台支持的协议和格式，不依赖浏览器 Cookie 或登录态。签名 URL 要控制有效期，令牌放在环境变量里，不写入仓库。

```http
POST {API_ENDPOINT}/os/v2/vlx/vr/stream
Authorization: Bearer {OM_API_KEY}
Content-Type: application/json
Accept: text/event-stream
```

`{API_ENDPOINT}` 以开发者平台「鉴权信息」页展示的地址为准。`model` 固定为 `vlx-vr`。`messages` 使用一条 `user` 消息，`content` 里放一段问题文本和一个 `video_url`。一次请求最多一个视频。`stream` 建议为 `true`。`temperature` 范围是 0 到 2，默认 0.7。需要调节推理强度时打开 `thinking`，再设置 `reasoning_effort`：`low`、`medium`、`high`、`max`，默认 `medium`。

返回是 SSE。同一请求的各帧共用首帧里的 `id`。推理过程增量在 `choices[0].delta.reasoning_content`，它只说明分析过程。业务结论取 `choices[0].delta.content` 并拼接。内容输出完成后会出现空的 `choices`，最后是 `data: [DONE]`。用户取消或超时不再需要结果时，用这个 `id` 调用 `POST {API_ENDPOINT}/os/v2/vlx/vr/stop`。

本地试验代码放在 `VLX-VR/`。

### 用 agent 和专用工具提高精度

一个模型同时猜动作、球队、球衣号和坐标，错了也无法单独重做。后期改成一个 agent 按事件调用专用工具：每件工具只回答一个问题，结果对不上就只重跑那一件，最后再按说明书校验。

agent 不直接填写 JSON。它拿到定位时间点后，按下面的顺序要结果，并把每次调用的片段时间和依据留下来。

| 工具 | 输入 | 只负责的结果 | 用来纠什么错 |
|---|---|---|---|
| 定位 | 整段视频 | `position_ms`、英文类别、置信度 | 时间点以 OpenSportsLib 为准，视觉模型不改时间 |
| 切片段 | `position_ms` 和相邻事件 | 原帧率片段 | 让后续工具看同一段画面 |
| 动作判别 | 片段，外加定位类别 | 说明书里的一个动作名 | 头球、出界、传中被收成传球时，用定位类别和画面分开复核 |
| 球衣颜色 | 动作帧上的行动人区域 | Team A 或 Team B，或无法判断 | 颜色来自这场比赛的元数据，不再把红白写死 |
| 球衣号 | 行动人躯干裁切 | `player_code`，看不清则为 `unknown` | 号和动作分开读，避免整段描述里的数字被当成球衣号 |
| 画面坐标 | 动作发生那一帧 | `player_screen_xy` | 用检测框中心，不用模型随口给的相对坐标 |
| 追踪关联 | 时间点加已对齐的追踪表 | `player_id`、`receiver_id` | 有追踪表才填跟踪号；没有就保持 `unknown` |
| 时序核对 | 按时间排好的相邻记录 | 传球人与下一脚触球人是否为同一跟踪号 | 接球、抢断、助攻和扑救的成对关系不一致时打回 |
| 整段推理 | 有争议的一段时间的视频 URL | VLX-VR 的结论和关键时间点 | 只在单片段工具互相矛盾时调用，用来看事件怎么发展 |
| 说明书校验 | 候选记录 | 字段顺序、动作名、帧序、`unknown` 规则 | 不通过就不交付，并指明该重跑哪一个工具 |

一轮里动作判别和定位类别冲突、球衣颜色没有结果、或时序核对失败，agent 只重跑对应工具。同一条事件重跑仍不通过，就整段交给 VLX-VR，并带上已有的时间点作为证据。VLX-VR 返回的是可回看的分析，agent 再把它拆回上表里的字段。拆不开的字段保持 `unknown`，记录进入人工复核。

这样精度来自工具分工和互相核对，而不是再换一个更大的提示词。角球、扑救、助攻在定位权重里没有对应类别，动作判别工具对这三类单独收样本，确认稳定之前一律复核。
