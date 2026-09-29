#!/usr/bin/env python3
"""Timed velocity pulses for DJI advanced virtual stick characterisation.

Publishes TwistStamped on <ns>/command/vel at a fixed rate, stepping through a
schedule of constant-velocity segments separated by zero dwells.

The zero dwells are not padding. They are the hold-quality measurement, and
they return the aircraft to a settled state before the next pulse.

Usage
-----
    python3 pulse_test.py --axis vx
    python3 pulse_test.py --axis yaw --amp 15.0 --pulse 2.0
    python3 pulse_test.py --axis vx --amp 0.5 --pulse 1.5   # once signs are known

Axis names: vx (forward), vy (lateral), vz (vertical), yaw (yaw rate).
Amplitudes are m/s for vx/vy/vz and deg/s for yaw.

On Ctrl-C the node publishes zeros briefly and exits. Dropping the stream
entirely is also safe: the app's 200 ms watchdog commands hover.
"""

import argparse
import math
import sys

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

# Hard ceilings
ABS_MAX_LINEAR = 0.6      # m/s
ABS_MAX_YAW = 30.0        # deg/s


class PulseTest(Node):

    def __init__(self, axis, amps, pulse_s, dwell_s, repeats, rate_hz, topic):
        super().__init__('pulse_test')

        self.pub = self.create_publisher(TwistStamped, topic, 10)
        self.rate_hz = rate_hz
        self.axis = axis

        # Schedule: list of (duration_s, value). Value applies to `axis` only.
        seq = [(dwell_s, 0.0)]
        for amp in amps:
            for _ in range(repeats):
                seq += [(pulse_s, +amp), (dwell_s, 0.0),
                        (pulse_s, -amp), (dwell_s, 0.0)]

        # Flatten to one value per tick so the timer stays trivial.
        self.plan = []
        self.labels = []
        for dur, val in seq:
            n = max(1, int(round(dur * rate_hz)))
            self.plan += [val] * n
            self.labels += [(val, dur)] + [None] * (n - 1)

        self.i = 0
        self.timer = self.create_timer(1.0 / rate_hz, self._tick)

        total = len(self.plan) / rate_hz
        self.get_logger().info(
            f"axis={axis} amps={amps} pulse={pulse_s}s dwell={dwell_s}s "
            f"repeats={repeats} -> {total:.1f}s on {topic}")

    def _msg(self, value):
        m = TwistStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'base_link'
        if self.axis == 'vx':
            m.twist.linear.x = value
        elif self.axis == 'vy':
            m.twist.linear.y = value
        elif self.axis == 'vz':
            m.twist.linear.z = value
        elif self.axis == 'yaw':
            m.twist.angular.z = math.radians(value)   # node expects rad/s
        return m

    def _tick(self):
        if self.i >= len(self.plan):
            self.get_logger().info("schedule complete, publishing zeros")
            self.pub.publish(self._msg(0.0))
            self.timer.cancel()
            rclpy.shutdown()
            return

        if self.labels[self.i] is not None:
            val, dur = self.labels[self.i]
            t = self.i / self.rate_hz
            self.get_logger().info(f"t={t:5.1f}s  {self.axis}={val:+.3f}  for {dur:.1f}s")

        self.pub.publish(self._msg(self.plan[self.i]))
        self.i += 1

    def stop(self):
        """Publish a short burst of zeros on the way out."""
        for _ in range(5):
            self.pub.publish(self._msg(0.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--axis', required=True, choices=['vx', 'vy', 'vz', 'yaw'])
    p.add_argument('--amp', type=float, default=None,
                   help='m/s for vx/vy/vz, deg/s for yaw')
    p.add_argument('--amps', default=None,
                   help='comma list, e.g. 0.1,0.2,0.3,0.4,0.5 -- sweeps in one run')
    p.add_argument('--pulse', type=float, default=1.5, help='pulse duration, s')
    p.add_argument('--dwell', type=float, default=3.0, help='zero dwell, s')
    p.add_argument('--repeats', type=int, default=2, help='+/- pairs')
    p.add_argument('--rate', type=float, default=20.0, help='publish rate, Hz')
    p.add_argument('--topic', default='/drone_1/command/vel')
    a = p.parse_args()

    if a.amps:
        amps = [float(x) for x in a.amps.split(',')]
    elif a.amp is not None:
        amps = [a.amp]
    else:
        amps = [15.0] if a.axis == 'yaw' else [0.2]

    limit = ABS_MAX_YAW if a.axis == 'yaw' else ABS_MAX_LINEAR
    bad = [x for x in amps if abs(x) > limit]
    if bad:
        print(f"refusing {bad}, hard limit for {a.axis} is {limit}")
        sys.exit(1)

    rclpy.init()
    node = PulseTest(a.axis, amps, a.pulse, a.dwell, a.repeats, a.rate, a.topic)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
