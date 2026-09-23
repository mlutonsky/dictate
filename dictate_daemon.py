#!/usr/bin/env python3
"""Dictation daemon (Canary-1B-v2). Send SIGUSR1 to toggle recording.

The model is preloaded into VRAM; /tmp/dictate_lang selects the language.

  canary = AED, translates: output language follows /tmp/dictate_lang (target_language).
"""

import signal
import os
import subprocess
import threading
import traceback
import datetime
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import scipy.signal as resampler

DAEMON_PID_FILE = "/tmp/dictate_daemon.pid"
WAV_FILE = "/tmp/dictate_audio.wav"
LOG_FILE = os.path.expanduser("~/.local/share/dictate/daemon.log")
MUTED_SINK_FILE = "/tmp/dictate_muted_sink"
RECORD_RATE = 48000
TARGET_RATE = 16000
DEVICE = "pipewire"
SCRIPT_NAME = "dictate_daemon.py"
LANG_FILE = "/tmp/dictate_lang"
SUPPORTED_LANGS = {"cs", "en", "de", "fr", "sk", "es"}
ONNX_MODEL_ID = "nemo-canary-1b-v2"

model = None
recording = False
audio_chunks = []
stream = None
current_lang = "cs"
toggle_event = threading.Event()


def log(msg):
    line = f"{datetime.datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def iter_sessions(obj, rt, _depth=0):
    """Yield (path, InferenceSession) for sessions held by a loaded onnx_asr model."""
    if _depth > 3:
        return
    for name, value in vars(obj).items():
        if isinstance(value, rt.InferenceSession):
            yield name, value
        elif hasattr(value, "__dict__"):
            for path, sess in iter_sessions(value, rt, _depth + 1):
                yield f"{name}.{path}", sess


def assert_on_gpu(model, rt):
    """Fail loudly if the model silently landed on CPU.

    get_available_providers() lists CUDAExecutionProvider even when its .so can't be
    loaded (missing/mismatched CUDA libs — e.g. a distro upgrade swapping CUDA 12 for 13).
    The session's own get_providers() is the only honest answer, so check that instead:
    CPU transcription is ~5x slower and otherwise fails silently.
    """
    sessions = list(iter_sessions(model, rt))
    if not sessions:
        log("WARNING: found no sessions on canary — cannot verify GPU use.")
        return
    on_cpu = [path for path, s in sessions if "CUDAExecutionProvider" not in s.get_providers()]
    if on_cpu:
        log(f"FATAL: canary is running on CPU, not GPU (sessions: {', '.join(on_cpu)}).")
        log("       CUDA provider failed to load — check LD_LIBRARY_PATH and the")
        log("       nvidia-*-cu12 wheels in the venv (ldd libonnxruntime_providers_cuda.so).")
        notify("Dictation FAILED", "canary fell back to CPU — GPU unavailable. Daemon stopped.")
        raise SystemExit(1)
    log(f"canary on GPU ({len(sessions)} sessions).")


def load_model():
    global model
    import onnx_asr
    import onnxruntime as rt
    # We don't have TensorRT (libnvinfer) installed and don't need it — dropping it
    # avoids the noisy "Failed to load libonnxruntime_providers_tensorrt.so" probe.
    # CUDA stays first (GPU), CPU is the fallback (used for the preprocessor/resampler).
    providers = [p for p in rt.get_available_providers() if p != "TensorrtExecutionProvider"]
    log(f"Providers: {providers}")
    log(f"Loading canary ({ONNX_MODEL_ID})...")
    notify("Dictation", "Loading canary model, please wait...")
    model = onnx_asr.load_model(ONNX_MODEL_ID, providers=providers)
    assert_on_gpu(model, rt)
    log("Model ready.")
    notify("Dictation", "Model ready (canary). Use Alt+, to start.")


def notify(title, body):
    subprocess.run(["notify-send", "-t", "3000", title, body], check=False)


def audio_callback(indata, frames, time_info, status):
    if recording:
        audio_chunks.append(indata.copy())


def handle_toggle(signum, frame):
    toggle_event.set()


def mute_output():
    """Mute the default audio sink; remember it for unmute.

    Muting (not pausing players) keeps live streams running, and leaves the volume
    itself untouched. A sink that was already muted is left alone — and not recorded,
    so unmute_output() won't unmute something the user muted.
    """
    try:
        sink = subprocess.run(
            ["pactl", "get-default-sink"], capture_output=True, text=True, check=False
        ).stdout.strip()
    except FileNotFoundError:
        return
    if not sink:
        return
    muted = subprocess.run(
        ["pactl", "get-sink-mute", sink], capture_output=True, text=True, check=False
    ).stdout.strip()
    if muted != "Mute: no":
        return
    subprocess.run(["pactl", "set-sink-mute", sink, "1"], check=False)
    with open(MUTED_SINK_FILE, "w") as f:
        f.write(sink)
    log(f"Muted sink: {sink}")


def unmute_output():
    """Unmute only the sink we muted."""
    try:
        with open(MUTED_SINK_FILE) as f:
            sink = f.read().strip()
        os.remove(MUTED_SINK_FILE)
    except FileNotFoundError:
        return
    subprocess.run(["pactl", "set-sink-mute", sink, "0"], check=False)
    log(f"Unmuted sink: {sink}")


def _read_lang() -> str:
    try:
        lang = open(LANG_FILE).read().strip()
        if lang in SUPPORTED_LANGS:
            return lang
    except Exception:
        pass
    return "cs"


def _do_toggle():
    global recording, audio_chunks, stream, current_lang

    log(f"Toggle received, recording={recording}")

    if not recording:
        current_lang = _read_lang()
        mute_output()
        audio_chunks = []
        recording = True
        stream = sd.InputStream(
            device=DEVICE,
            samplerate=RECORD_RATE,
            channels=1,
            dtype="int16",
            callback=audio_callback,
        )
        stream.start()
        log(f"Recording started (lang={current_lang}).")
        notify("Dictation", f"Recording · CANARY → {current_lang.upper()}")
    else:
        recording = False
        stream.stop()
        stream.close()
        stream = None
        unmute_output()

        if not audio_chunks:
            log("No audio captured.")
            notify("Dictation", "No audio captured.")
            return

        log(f"Stopped. Got {len(audio_chunks)} chunks. Resampling...")
        audio = np.concatenate(audio_chunks, axis=0)
        audio_16k = resampler.resample_poly(audio, TARGET_RATE, RECORD_RATE).astype(np.int16)
        wav.write(WAV_FILE, TARGET_RATE, audio_16k)

        log(f"Transcribing (lang={current_lang})...")
        result = model.recognize(WAV_FILE, language=current_lang, target_language=current_lang)
        log(f"Result: {result!r}")

        if result and result.strip():
            # Don't touch the clipboard here — the ydotool wrapper owns it. It saves the
            # clipboard, pastes, and restores; copying the transcript first would just make
            # it save (and restore) the transcript, defeating the whole thing.
            r = subprocess.run(["ydotool", "type", result.strip()])
            log(f"ydotool type done: rc={r.returncode}")
            if r.returncode != 0:
                log(f"ydotool error: rc={r.returncode}")
        else:
            log("Nothing recognized.")
            notify("Dictation", "Nothing recognized.")


def main_loop():
    global recording, stream

    while True:
        toggle_event.wait()
        toggle_event.clear()
        try:
            _do_toggle()
        except BaseException as e:
            log(f"ERROR: {type(e).__name__}: {e}")
            log(traceback.format_exc())
            notify("Dictation", f"Error: {e}")
            recording = False
            if stream:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
                stream = None
            unmute_output()


def watchdog_loop():
    """Restart main_loop thread if it ever dies."""
    while True:
        t = threading.Thread(target=main_loop, daemon=True, name="main_loop")
        t.start()
        t.join()
        log("WARNING: main_loop thread died — restarting.")


def _is_daemon_cmdline(cmdline: str) -> bool:
    """True only for a python process actually running this script.

    A bare substring match also catches processes that merely *mention* the
    script — `pgrep -f dictate_daemon.py`, a grep, the shell running either —
    and the daemon then refuses to start, claiming it is already up. So split
    the NUL-separated argv and require the script to be an argument of a python
    interpreter, or argv[0] itself when launched through the shebang.
    """
    argv = [a for a in cmdline.split("\0") if a]
    if not argv:
        return False
    if os.path.basename(argv[0]) == SCRIPT_NAME:
        return True
    if not os.path.basename(argv[0]).startswith("python"):
        return False
    return any(os.path.basename(a) == SCRIPT_NAME for a in argv[1:])


def already_running():
    """Return the PID of another live dictate daemon, else None.
    Scans /proc directly so it stays reliable even when the PID file was lost
    (crash, /tmp cleanup, or a previous failed launch removed it). The PID file
    is only a convenience for `dictate` to signal us, not the source of truth."""
    me = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == me:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmdline = f.read().decode("utf-8", "replace")
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if _is_daemon_cmdline(cmdline):
            return pid
    return None


if __name__ == "__main__":
    running_pid = already_running()
    if running_pid is not None:
        log(f"Daemon already running (PID {running_pid}) — exiting.")
        notify("Dictation", f"Daemon already running (PID {running_pid})")
        raise SystemExit(0)

    with open(DAEMON_PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    try:
        load_model()
        signal.signal(signal.SIGUSR1, handle_toggle)

        watchdog = threading.Thread(target=watchdog_loop, daemon=True, name="watchdog")
        watchdog.start()

        log(f"Dictation daemon ready (PID {os.getpid()})")
        while True:
            signal.pause()
    finally:
        if os.path.exists(DAEMON_PID_FILE):
            os.remove(DAEMON_PID_FILE)
