#!/usr/bin/env python3
"""
Probe the WildBridge telemetry socket and report which fields refresh together.

Purpose: verify that speed, attitude, altitudeAgl and gimbalJointAttitude are
refreshed by the MSDK at the same rate. If they are, a single per-packet phone
timestamp is valid. If the gimbal (or anything else) refreshes faster, a
change-detector keyed on it would stamp stale values with fresh timestamps.

Usage:
    ./telemetry_probe.py --host 192.168.50.18 --port 8081
    ./telemetry_probe.py --mode poll --hz 20          # request/response servers
    ./telemetry_probe.py --log run1.jsonl             # also save raw samples

Ctrl-C to stop and print the summary.
"""

import argparse
import json
import signal
import socket
import sys
import time
from collections import Counter

FIELDS = [
    "speed",
    "attitude",
    "relativeAltitude",
    "gimbalJointAttitude",
    "pktTMonoNs",
    "gimbalTMonoNs",
]

_stop = False


def _on_sigint(signum, frame):
    global _stop
    _stop = True


def iter_push(sock):
    """Server streams JSON objects back to back or newline-delimited."""
    dec = json.JSONDecoder()
    buf = ""
    while not _stop:
        chunk = sock.recv(65536)
        if not chunk:
            return
        buf += chunk.decode("utf-8", "replace")
        while True:
            buf = buf.lstrip()
            if not buf:
                break
            try:
                obj, end = dec.raw_decode(buf)
            except json.JSONDecodeError:
                break  # partial object, wait for more bytes
            buf = buf[end:]
            yield time.monotonic(), obj


def iter_poll(host, port, hz, timeout):
    """Server returns one object per request."""
    dec = json.JSONDecoder()
    period = 1.0 / hz
    while not _stop:
        t0 = time.monotonic()
        try:
            with socket.create_connection((host, port), timeout=timeout) as s:
                s.sendall(b"\n")
                buf = b""
                while True:
                    chunk = s.recv(65536)
                    if not chunk:
                        break
                    buf += chunk
                    try:
                        obj, _ = dec.raw_decode(buf.decode("utf-8", "replace").lstrip())
                    except json.JSONDecodeError:
                        continue
                    yield time.monotonic(), obj
                    break
        except (OSError, socket.timeout) as e:
            print(f"poll error: {e}", file=sys.stderr)
        sleep = period - (time.monotonic() - t0)
        if sleep > 0:
            time.sleep(sleep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.50.18")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--mode", choices=["push", "poll"], default="push")
    ap.add_argument("--hz", type=float, default=20.0, help="poll rate (poll mode)")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--log", help="append raw samples as JSONL")
    args = ap.parse_args()

    signal.signal(signal.SIGINT, _on_sigint)

    logf = open(args.log, "a") if args.log else None
    last = {}
    change_counts = Counter()
    change_sets = Counter()
    last_change_t = {}
    intervals = {f: [] for f in FIELDS}
    n = 0
    t_start = None
    t_last = None

    if args.mode == "push":
        sock = socket.create_connection((args.host, args.port), timeout=args.timeout)
        print(f"connected to {args.host}:{args.port} (push mode)", file=sys.stderr)
        source = iter_push(sock)
    else:
        print(f"polling {args.host}:{args.port} at {args.hz} Hz", file=sys.stderr)
        source = iter_poll(args.host, args.port, args.hz, args.timeout)

    try:
        for t, obj in source:
            n += 1
            if t_start is None:
                t_start = t
            t_last = t
            if logf:
                logf.write(json.dumps({"t": t, "d": obj}) + "\n")

            changed = []
            for f in FIELDS:
                v = obj.get(f)
                if f in last and v != last[f]:
                    changed.append(f)
                    change_counts[f] += 1
                    if f in last_change_t:
                        intervals[f].append(t - last_change_t[f])
                    last_change_t[f] = t
                elif f not in last:
                    last_change_t[f] = t
                last[f] = v

            if changed:
                change_sets[tuple(sorted(changed))] += 1
                print(f"[{t - t_start:7.3f}] {' '.join(changed)}")
    except (OSError, socket.timeout) as e:
        print(f"stream ended: {e}", file=sys.stderr)
    finally:
        if logf:
            logf.close()

    if not n or t_last is None or t_last <= t_start:
        print("\nno usable samples", file=sys.stderr)
        return

    span = t_last - t_start
    print("\n" + "=" * 62)
    print(f"samples: {n} over {span:.1f} s  ({n / span:.1f} Hz read rate)")
    print("=" * 62)

    print(f"\n{'field':24} {'changes':>8} {'rate Hz':>9} {'median dt ms':>13}")
    for f in FIELDS:
        c = change_counts[f]
        iv = sorted(intervals[f])
        med = iv[len(iv) // 2] * 1e3 if iv else float("nan")
        print(f"{f:24} {c:>8} {c / span:>9.2f} {med:>13.1f}")

    print("\nco-occurrence: P(B changed | A changed)")
    core = [f for f in FIELDS if not f.endswith("MonoNs")]
    hdr = "".join(f"{b[:11]:>13}" for b in core)
    print(f"{'':24}{hdr}")
    for a in core:
        row = f"{a:24}"
        for b in core:
            if change_counts[a] == 0:
                row += f"{'-':>13}"
                continue
            both = sum(v for k, v in change_sets.items() if a in k and b in k)
            row += f"{both / change_counts[a]:>13.2f}"
        print(row)

    print("\nchange patterns observed (most common first):")
    for combo, c in change_sets.most_common(12):
        names = " + ".join(x for x in combo if not x.endswith("MonoNs"))
        print(f"  {c:>6}  {names or '(stamps only)'}")

    print(
        "\nInterpretation:\n"
        "  Lockstep  -> off-diagonal values all ~1.00, one dominant change pattern\n"
        "               containing all four fields. Single packet stamp is valid.\n"
        "  Not       -> a field with a higher rate, or patterns where it changes\n"
        "               alone. Exclude it from the change key.\n"
        "  A field with 0 changes was not excited - see test conditions."
    )


if __name__ == "__main__":
    main()
    