#!/usr/bin/env python3
"""解析 TensorBoard 事件文件，提取标量并做简要分析（用于 phase1 vs phase1_续训 对比）。"""
import os
import sys

try:
    from tensorboard.backend.event_processing import event_accumulator
except ImportError:
    try:
        from tensorflow.summary.summary_iterator import summary_iterator
        HAS_TF_ITERATOR = True
    except ImportError:
        HAS_TF_ITERATOR = False
    summary_iterator = None
    event_accumulator = None

def load_with_ea(logdir, tag_filter=None):
    """使用 EventAccumulator 加载（tensorboard 包）。"""
    ea = event_accumulator.EventAccumulator(logdir)
    ea.Reload()
    out = {}
    for tag in ea.Tags().get("scalars", []):
        if tag_filter and not any(f in tag for f in tag_filter):
            continue
        events = ea.Scalars(tag)
        out[tag] = [(e.step, e.wall_time, e.value) for e in events]
    return out

def load_with_tf(logdir, tag_filter=None):
    """使用 summary_iterator 加载（tensorflow 包）。"""
    from tensorflow.core.util import event_pb2
    out = {}
    for root, _, files in os.walk(logdir):
        for f in sorted(files):
            if not f.startswith("events.out.tfevents"):
                continue
            path = os.path.join(root, f)
            try:
                for event in summary_iterator.summary_iterator(path):
                    if not event.summary.value:
                        continue
                    for v in event.summary.value:
                        if v.WhichOneof("value") != "simple_value":
                            continue
                        tag = v.tag
                        if tag_filter and not any(f in tag for f in tag_filter):
                            continue
                        if tag not in out:
                            out[tag] = []
                        out[tag].append((event.step, event.wall_time, v.simple_value))
            except Exception as e:
                print(f"  Warning: skip {path}: {e}", file=sys.stderr)
    for tag in out:
        out[tag].sort(key=lambda x: (x[0], x[1]))
    return out

def main():
    base = os.path.dirname(os.path.abspath(__file__))
    phase1_dir = os.path.join(base, "ppo_g1_logs", "phase1", "PPO_1")
    phase1_cont_dir = os.path.join(base, "ppo_g1_logs", "phase1_续训", "PPO_1")

    # 常用曲线 tag 前缀（SB3 常见）
    tag_filter = ["rollout/ep_rew", "train/", "time/", "rollout/ep_len", "robust", "domain_rand"]

    load_fn = None
    if event_accumulator is not None:
        load_fn = load_with_ea
    elif HAS_TF_ITERATOR and summary_iterator is not None:
        load_fn = load_with_tf
    else:
        print("需要安装 tensorboard 或 tensorflow 以解析事件文件。", file=sys.stderr)
        sys.exit(1)

    print("加载 phase1 (基础训练) ...")
    data_phase1 = load_fn(phase1_dir, tag_filter) if os.path.isdir(phase1_dir) else {}
    print("加载 phase1_续训 (继续训练) ...")
    data_cont = load_fn(phase1_cont_dir, tag_filter) if os.path.isdir(phase1_cont_dir) else {}
    # 续训完整 tag 列表（无过滤）
    data_cont_all = load_fn(phase1_cont_dir, None) if os.path.isdir(phase1_cont_dir) else {}
    if data_cont_all and set(data_cont_all) != set(data_cont):
        print("\n[续训] 全部标量 tag:", sorted(data_cont_all.keys()))

    def stats(name, points):
        if not points:
            return "无数据"
        steps = [p[0] for p in points]
        vals = [p[2] for p in points]
        return "step [%s, %s]  mean=%.4f min=%.4f max=%.4f n=%d" % (
            min(steps), max(steps), sum(vals)/len(vals), min(vals), max(vals), len(points)
        )

    print("\n========== phase1 (基础) ==========")
    for tag in sorted(data_phase1.keys()):
        print("  %s: %s" % (tag, stats(tag, data_phase1[tag])))

    print("\n========== phase1_续训 (继续训练) ==========")
    for tag in sorted(data_cont.keys()):
        print("  %s: %s" % (tag, stats(tag, data_cont[tag])))

    # 续训曲线分析：关键指标
    print("\n========== 续训曲线简要分析 ==========")
    key_tags = ["rollout/ep_rew_mean", "train/ep_rew_mean", "train/ent_coef", "train/learning_rate",
                "train/loss", "train/policy_gradient_loss", "train/value_loss", "time/fps"]
    for tag in key_tags:
        if tag in data_cont:
            pts = data_cont[tag]
            if len(pts) >= 2:
                v0, v1 = pts[0][2], pts[-1][2]
                trend = "上升" if v1 > v0 else ("下降" if v1 < v0 else "持平")
                print("  %s: 首=%.4f 末=%.4f -> %s (step %d -> %d)" % (tag, v0, v1, trend, pts[0][0], pts[-1][0]))
            else:
                print("  %s: %s" % (tag, stats(tag, pts)))
    # 若有 rollout 回报
    for tag in sorted(data_cont.keys()):
        if "rollout" in tag and "rew" in tag and "mean" in tag:
            pts = data_cont[tag]
            if len(pts) >= 2:
                v0, v1 = pts[0][2], pts[-1][2]
                trend = "上升" if v1 > v0 else ("下降" if v1 < v0 else "持平")
                print("  [续训] %s: 首=%.4f 末=%.4f -> %s" % (tag, v0, v1, trend))

    # phase1 末尾 vs 续训 开头（衔接性）
    print("\n========== 衔接性：phase1 末尾 vs 续训 开头 ==========")
    for tag in ["train/loss", "train/value_loss", "train/approx_kl", "train/learning_rate"]:
        if tag in data_phase1 and tag in data_cont:
            end_phase1 = data_phase1[tag][-1]
            start_cont = data_cont[tag][0]
            print("  %s: phase1末(step=%d)=%.4f  续训首(step=%d)=%.4f  差异=%.4f" % (
                tag, end_phase1[0], end_phase1[2], start_cont[0], start_cont[2], start_cont[2] - end_phase1[2]
            ))

if __name__ == "__main__":
    main()
