# FootballMatchPursuit

把一段足球转播视频写成 [事件说明书](事件说明书.md) 规定的事件 JSON。当前在一台 GTX 1060 6GB 上跑通了两条模型，先定位时间点，再看每个事件的视频片段补球队、动作、球衣号和画面位置。跟踪号在转播画面里看不见，`player_id` 和 `receiver_id` 仍写 `unknown`。

更完整的系统边界见 [足球视频事件JSON系统概要设计.md](足球视频事件JSON系统概要设计.md)。字段样例是 [test_data/test.json](test_data/test.json)，样例按 30 帧/秒书写。

## 用什么模型做了什么

| 步骤 | 模型 | 代码 | 做成了什么 | 做不到什么 |
|---|---|---|---|---|
| 动作定位 | [OpenSportsLab/OSL-loc-snbas-2025-e2e](https://huggingface.co/OpenSportsLab/OSL-loc-snbas-2025-e2e) | `main.py` | 从整段视频找出动作发生的毫秒，并给出英文类别。类别包括 PASS、DRIVE、HEADER、HIGH PASS、OUT、CROSS、THROW IN、SHOT、BALL PLAYER BLOCK、PLAYER SUCCESSFUL TACKLE、FREE KICK、GOAL | 不认识人，不写中文事件名，也不给画面坐标。角球、扑救、助攻不在这 12 类里 |
| 片段重判 | [Qwen/Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct) | `recognize_unknown_events.py` | 为每个定位时间点切出原帧率片段，让模型在说明书动作里选择，并读出球队、球衣号和身体中心坐标 | 跟踪号仍是 `unknown`。2B 模型在这块 6GB 显卡上经常把不同动作收成传球，球衣号大多读不出来 |

定位模型是 RegNetY-008 加 GRU 的端到端权重。它按每秒 2 帧抽帧，每 100 帧一个窗口。`infer()` 返回的是高召回候选；`main.py` 保存的是非极大值抑制之后的 `results_spotting_test.json`。

视觉模型看的是每个事件自己的片段：从与上一个事件的中点切到与下一个事件的中点，两边间隔很长时单侧最多 2 秒。片段文件保持原分辨率，模型再按每秒 2 帧抽帧，并把单次画面像素限制在 6GB 显存放得下的范围。提示词把红队当作 Team A、白队当作 Team B。这是当前转播的颜色约定，换一场比赛要改提示词。

`export_events.py` 是不调用视觉模型的对照脚本。它只把能直接对应的英文类别写成中文动作，人员、球队和坐标保持未知，头球、出界、传中等对不上的类别直接跳过。

## 现在怎么跑

环境是 conda 环境 `osl`（Python 3.12）。这台机器的驱动只报告 CUDA 12.2，官方 `opensportslib setup` 要求 CUDA 12.6，因此使用仍能在 Pascal 上执行的 PyTorch 2.5.1+cu121。未安装 DALI，视频由 OpenCV 解码。6GB 显存下定位模型的 batch size 固定为 1。

```bash
conda activate osl
python main.py --video test_data/test.mp4
python recognize_unknown_events.py
```

短流程可以只跑开头若干秒：

```bash
python main.py --video test_data/test.mp4 --max-seconds 80
```

主要产物：

| 文件 | 内容 |
|---|---|
| `outputs/opensportslib/<视频名>.predictions.json` | 定位结果。`position_ms` 是动作时刻，`frame` 是模型 2 帧/秒下的帧号 |
| `outputs/opensportslib/clips/vl/` | 每个事件的完整片段 |
| `outputs/opensportslib/test.events.json` | 视觉模型重判后的说明书记录。这里的 `frame` 按源视频帧率计算，`test.mp4` 为 30 帧/秒 |

视频文件用 Git LFS 存放。

## 当前结果还不能直接交付

定位结果没有行动人。视觉模型补上的球队、动作和坐标是候选，不是复核后的交付件。2B 模型会把不少头球、出界和传中也判成传球，坐标比较粗，球衣号经常是 `unknown`。

说明书还要求传球和接球人用同一个跟踪号串起来，抢断、射门、助攻、扑救也有成对关系。这些关系现在都还没有。

## 后期调研：VLX-VR

本地 Qwen3-VL-2B 只能看几秒长的事件片段，整场比赛的跨片段关系仍然串不起来。下一步调研 [VLX-VR](VLX-VR/)（Video Reasoning），用它做允许等待的整段视频语义分析。需要持续采样、快速反馈或实时告警时，评估同系列的 VLX-Flow。

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
