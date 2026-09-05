"""Plot measured stream length, active tokens and sampled process memory."""

import argparse
import bisect
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    events = [json.loads(line) for line in (args.result/'events.jsonl').read_text().splitlines()]
    events = [row for row in events if 'elapsed_s' in row]
    samples = [json.loads(line) for line in (args.result/'resources.jsonl').read_text().splitlines()]
    times = [row['elapsed_s'] for row in events]
    xs, gpu, server, client = [], [], [], []
    for sample in samples:
        index = bisect.bisect_right(times, sample['elapsed_s']) - 1
        if index < 0 or sample['elapsed_s'] > times[-1] or 'server_gpu_mib' not in sample:
            continue
        xs.append(events[index]['total_input_tokens'])
        gpu.append(sample['server_gpu_mib'])
        server.append(sample['server_rss_kib']/1024)
        client.append(sample['client_rss_kib']/1024)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    axes[0].plot([r['total_input_tokens'] for r in events], [r['active_tokens'] for r in events], color='#2364aa')
    axes[0].set(ylabel='Active tokens', ylim=(0, 560), title='512-token configured window')
    axes[1].plot(xs, gpu, label='GPU process', color='#2364aa')
    axes[1].plot(xs, server, label='Server RSS', color='#cd5c2e')
    axes[1].plot(xs, client, label='Client RSS', color='#31856a')
    axes[1].set(ylabel='Sampled memory (MiB)', ylim=(0, 1100), title='Qwen3.5-0.8B Q8_0; RTX 2070 SUPER')
    axes[1].legend(frameon=False, fontsize=8)
    for axis in axes:
        axis.set_xlabel('Cumulative input tokens')
        axis.ticklabel_format(axis='x', style='sci', scilimits=(6, 6))
        axis.grid(alpha=0.15)
        axis.spines[['top', 'right']].set_visible(False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, metadata={'Date': None} if args.output.suffix == '.svg' else {})
    print(args.output)


if __name__ == '__main__':
    main()
