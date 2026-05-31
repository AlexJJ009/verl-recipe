Start and supervise the staged-v1 Stage 1 WxPusher completion monitor.

Run this command and keep it running:

```bash
python3 recipe/on_policy_wdl_sft/staged_v1/monitor_stage1_wxpusher.py --loop --poll-sec 120 --final-step 150
```

Rules:
- Do not launch training.
- Do not delete checkpoints, Ray files, W&B files, logs, or temporary files.
- Send notifications only through the monitor script when a Stage 1 beta run reaches final step.
- The monitor writes the exact short message to `recipe/on_policy_wdl_sft/staged_v1/monitor_notifications/messages/` before sending.
- If the monitor exits because all beta runs are complete or because of an error, report the concise reason in the final message.
