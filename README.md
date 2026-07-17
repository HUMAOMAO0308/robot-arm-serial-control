# Robot Arm Serial Control

This small project wraps the original one-off script into a reusable command line tool.

## Features

- configurable COM port, timing, speed, and initial joint angles
- safer cleanup that always tries to return to the initial pose
- clearer command logging for each movement step

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python main.py --port COM3
```

Example with custom sweep settings:

```bash
python main.py --port COM3 --j3 150 --step-angle 0.1 --total-steps 200 --pause-time 0.5
```

Scan and capture one ZWO image during each pause:

```bash
python scan_with_zwo_capture.py --port COM3 --pause-time 1.5 --capture-delay 0.4 --output-dir captures
```

Capture outputs:

- one PNG image per step
- `captures_log.csv` with timestamp, step index, joint angles, timing, and image path
- `captures_summary.json` with the full run configuration and all capture records

## Notes

- The script sends `!START` first, waits for homing, then moves to the initial pose.
- During shutdown, it always tries to return to the initial pose before closing the serial port.
- Verify the joint direction on your hardware before increasing speed or step count.
- `scan_with_zwo_capture.py` saves one image per step and supports `--disable` if you want to send `!DISABLE` on exit.
