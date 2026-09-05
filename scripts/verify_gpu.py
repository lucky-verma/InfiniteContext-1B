"""Check a CPU or CUDA execution path; exit nonzero when it is unavailable."""

import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    args = parser.parse_args()
    report = {'python': sys.version.split()[0], 'requested_device': args.device}
    try:
        import torch
        report['torch'] = torch.__version__
        report['cuda_available'] = torch.cuda.is_available()
        if args.device == 'cuda' and not report['cuda_available']:
            raise RuntimeError('CUDA was requested but is unavailable')
        value = torch.tensor([[1., 2.], [3., 4.]], device=args.device)
        torch.testing.assert_close(value @ value, torch.tensor([[7., 10.], [15., 22.]], device=args.device))
        if args.device == 'cuda':
            properties = torch.cuda.get_device_properties(0)
            report.update(gpu=properties.name, memory_bytes=properties.total_memory,
                          compute_capability=list(torch.cuda.get_device_capability(0)))
        report['status'] = 'passed'
    except (ImportError, RuntimeError, AssertionError) as error:
        report.update(status='failed', error=str(error))
    print(json.dumps(report, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
