import argparse
import json
import os
import signal
import time
from pathlib import Path

from pymavlink import mavutil


running = True


def stop(signum, frame):
    global running
    running = False


def save_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(state, ensure_ascii=False),
        encoding="utf-8",
    )
    os.replace(temporary_path, path)


parser = argparse.ArgumentParser()
parser.add_argument("--device", default="/dev/ttyACM0")
parser.add_argument("--baud", type=int, default=115200)
parser.add_argument(
    "--output",
    default="/home/raspberrypiuser/latest_state.json",
)
args = parser.parse_args()

output_path = Path(args.output)

state = {
    "updated_at": 0.0,
    "connected": False,
    "system_id": None,
    "component_id": None,
    "lat": None,
    "lon": None,
    "relative_alt_m": None,
    "vx_m_s": None,
    "vy_m_s": None,
    "vz_m_s": None,
    "roll_rad": None,
    "pitch_rad": None,
    "yaw_rad": None,
    "gps_fix": None,
    "satellites": None,
    "battery_v": None,
}

signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

while running:
    try:
        print(f"Connecting to MAVLink on {args.device}", flush=True)

        mav = mavutil.mavlink_connection(
            args.device,
            baud=args.baud,
            autoreconnect=True,
            source_system=255,
        )

        heartbeat_deadline = time.monotonic() + 15
        heartbeat = None

        while running and time.monotonic() < heartbeat_deadline:
            message = mav.recv_match(blocking=True, timeout=1)

            if message is not None and message.get_type() == "HEARTBEAT":
                heartbeat = message
                break

        if heartbeat is None:
            raise ConnectionError("No MAVLink HEARTBEAT received")

        state["connected"] = True
        state["system_id"] = heartbeat.get_srcSystem()
        state["component_id"] = heartbeat.get_srcComponent()
        save_state(output_path, state)

        print(
            f"Connected: system={state['system_id']}, "
            f"component={state['component_id']}",
            flush=True,
        )

        last_message_at = time.monotonic()
        last_save_at = 0.0

        while running:
            message = mav.recv_match(blocking=True, timeout=1)
            now = time.time()

            if message is None:
                if time.monotonic() - last_message_at > 10:
                    raise ConnectionError("MAVLink stream timed out")
                continue

            last_message_at = time.monotonic()
            message_type = message.get_type()

            if message_type == "GLOBAL_POSITION_INT":
                state["updated_at"] = now
                state["lat"] = message.lat / 1e7
                state["lon"] = message.lon / 1e7
                state["relative_alt_m"] = message.relative_alt / 1000.0
                state["vx_m_s"] = message.vx / 100.0
                state["vy_m_s"] = message.vy / 100.0
                state["vz_m_s"] = message.vz / 100.0

            elif message_type == "GPS_RAW_INT":
                state["gps_fix"] = message.fix_type
                state["satellites"] = message.satellites_visible

            elif message_type == "ATTITUDE":
                state["roll_rad"] = message.roll
                state["pitch_rad"] = message.pitch
                state["yaw_rad"] = message.yaw

            elif message_type == "SYS_STATUS":
                if message.voltage_battery != 65535:
                    state["battery_v"] = message.voltage_battery / 1000.0

            if now - last_save_at >= 0.2:
                save_state(output_path, state)
                last_save_at = now

    except Exception as error:
        state["connected"] = False
        save_state(output_path, state)
        print(f"MAVLink error: {error}; retrying in 3 seconds", flush=True)
        time.sleep(3)

state["connected"] = False
save_state(output_path, state)
print("MAVLink reader stopped", flush=True)
