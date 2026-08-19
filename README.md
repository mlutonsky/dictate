# dictate

Local, offline speech-to-text for Linux desktops. Hold a keyboard shortcut, speak, and the
transcript is typed into whatever window has focus. Runs entirely on your own GPU — no cloud,
no API keys, no audio leaves the machine.

Czech in, English out (or vice versa) works too: the Canary model translates as it transcribes.

## What it does

One NVIDIA speech model stays resident in VRAM behind a small daemon:

| Model | Type | What it's for |
|---|---|---|
| `canary` (nemo-canary-1b-v2) | AED | **Translates.** Output language follows the shortcut you press — speak Czech, hit `Alt+.`, get English. |

It is the ONNX port by [istupakov](https://huggingface.co/istupakov), driven through
[`onnx_asr`](https://github.com/istupakov/onnx-asr) on onnxruntime-gpu. NeMo itself is not used.

It occupies ~5.3 GB of VRAM. Transcription runs roughly 25x faster than realtime.

## How it works

```
Alt+, ──> dictate cs ──┐
Alt+. ──> dictate en ──┼──> writes /tmp/dictate_lang, sends SIGUSR1 ──> dictate_daemon.py
                                                                            │
                    record (PipeWire, 48kHz) ──> resample to 16kHz ─────────┤
                    transcribe on GPU ──> ydotool ──> text into focused window
```

The daemon is a single long-running process started at login. It has no IPC beyond signals:

- **`SIGUSR1` toggles recording.** First press starts, second stops and transcribes.
- **`/tmp/dictate_lang`** holds the target language. It is read at the *start of a recording*, so
  switching languages never interrupts one in flight.
- **`/tmp/dictate_daemon.pid`** is how the shortcuts find the daemon.
- Media players are paused during recording and resumed afterwards (via `playerctl`).

Keeping the model preloaded is the whole point: model load takes ~5s, so loading on demand would
make every dictation unusable. The cost is the VRAM; `dictate-stop` reclaims it.

## Commands

| Command | Does |
|---|---|
| `dictate [cs\|en\|de\|fr\|sk\|es]` | Toggle recording; the argument is the **target** language (default `cs`). |
| `dictate-daemon` | Start the daemon in the foreground. |
| `dictate-stop` | Stop it and free the VRAM. |
| `dictate-restart` | Stop, relaunch detached, reload the model. |

Default shortcuts: `Alt+,` Czech · `Alt+.` English.

## Requirements

- **NVIDIA GPU with ~5.5 GB free VRAM** and a driver supporting CUDA 12.
- **Python 3.12** — the venv is built with [`uv`](https://github.com/astral-sh/uv).
- **PipeWire** for capture, **ydotool** for typing, **playerctl** for the pause/resume, **wl-copy**
  (see the ydotool section below), `notify-send` for the desktop notifications.
- ~3.7 GB of disk for the model, cached in `~/.cache/huggingface` on first run.

CUDA itself is **not** required system-wide — every CUDA library is pulled into the venv as an
`nvidia-*-cu12` wheel. That is deliberate; see Gotchas.

## Install

The repo is the source of truth; the live locations are symlinks into it. Clone anywhere
(`~/bin/dictate` here) and wire it up:

```bash
# 1. venv + dependencies
uv venv --python 3.12 ~/.local/share/dictate/venv
uv pip install --python ~/.local/share/dictate/venv/bin/python -r requirements.txt

# 2. symlink the scripts and the daemon into place
mkdir -p ~/.local/bin ~/.local/share/dictate
for f in dictate dictate-daemon dictate-restart dictate-stop; do
    ln -sf "$PWD/bin/$f" ~/.local/bin/$f
done
ln -sf "$PWD/dictate_daemon.py" ~/.local/share/dictate/dictate_daemon.py

# 3. autostart at login (.desktop files can't expand ~, so the path is baked in)
sed "s|/home/YOUR_USER|$HOME|" autostart/dictate-daemon.desktop > ~/.config/autostart/dictate-daemon.desktop

# 4. ydotool daemon (needed to type text)
cp systemd/ydotoold.service ~/.config/systemd/user/
systemctl --user enable --now ydotoold
sudo cp ydotool-clipboard-wrapper.sh /usr/local/bin/ydotool   # see caveat below

# 5. GNOME shortcuts — bind these commands to Alt+, and Alt+.
#    Settings > Keyboard > Custom Shortcuts:
#      ~/.local/bin/dictate cs        Alt+comma
#      ~/.local/bin/dictate en        Alt+period
```

First `dictate-daemon` run downloads the model (~3.7 GB) and takes a few minutes.

## Text insertion, and the ydotool wrapper

`ydotool-clipboard-wrapper.sh` is installed as `/usr/local/bin/ydotool`, **shadowing the real
`/usr/bin/ydotool`** in PATH. It intercepts `ydotool type` and, instead of typing the text
keystroke by keystroke, pipes it into `wl-copy` and presses Ctrl+V. Everything else is passed
through to the real binary.

**This is not a workaround waiting to be removed — on GNOME/Wayland it is the only thing that
works.** Both alternatives are dead ends:

- `ydotool type` emits **raw US keycodes** ("there's no way to know how many keyboard layouts are
  there in the world, we're using raw keycodes now" — its own help). Czech diacritics have no US
  keycode, so they are silently dropped; and with a `cz+qwerty` layout active even ASCII comes out
  scrambled, because the app decodes the keycodes through whatever layout is current.
- `wtype` builds its own keymap and would handle any character, but it needs the
  `virtual-keyboard` protocol, which Mutter does not implement: *"Compositor does not support the
  virtual keyboard protocol"*.

The wrapper saves the clipboard before pasting and restores it afterwards, so dictation no longer
destroys what you had copied. What to know:

- **Restore is timing-based, not synchronised.** After Ctrl+V the app fetches the data within
  milliseconds; the wrapper waits `YDOTOOL_CLIP_RESTORE_DELAY` (default 0.5s) before restoring —
  a ~100x margin. If it were ever too short, the app would read the *restored* content and paste
  the wrong text. Proper synchronisation is impossible here: `wl-copy --paste-once` would say
  exactly when the paste happened, but GNOME's clipboard manager reads the selection itself and
  consumes that single serve immediately.
- **One MIME type is restored**, preferring UTF-8 text; `image/*` survives too. Multi-type offers
  (PhpStorm publishes a Java cookie plus a dozen encodings) can't be reproduced faithfully.
- It requires `wl-copy`/`wl-paste` and a focused window that accepts Ctrl+V.
- It shadows a system binary from `/usr/local/bin`, which is surprising to anyone debugging this
  later — including you. If dictation records fine but no text appears, look here first.

## Gotchas

**onnxruntime-gpu needs CUDA 12, not whatever the distro ships.** The wheel is built against
CUDA 12 and needs `libcudart.so.12`, `libcufft.so.11`, `libcublasLt.so.12`, `libcudnn.so.9`. When
Fedora 44 replaced system CUDA 12.6 with 13.x, the sonames stopped matching, the CUDA provider
failed to load, and everything silently fell back to CPU — ~5x realtime instead of ~25x, with no
error anywhere. Hence: all CUDA libs live in the venv as `nvidia-*-cu12` wheels, and
`bin/dictate-daemon` globs every `nvidia/*/lib` into `LD_LIBRARY_PATH`. A distro CUDA upgrade
can no longer touch this.

**The CPU fallback is silent, so the daemon checks for it.** `get_available_providers()` reports
`CUDAExecutionProvider` even when its `.so` cannot be loaded, so logs look healthy while the GPU
sits idle. onnxruntime does log the real error — but on native stderr, which autostart sends to
`/dev/null`. Only a session's own `get_providers()` tells the truth, so `assert_on_gpu()` checks
every session after load and hard-exits with a desktop notification rather than running slowly.

Verify by hand any time:

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv   # expect ~5.3 GB
```

**Do not try to fail fast by dropping `CPUExecutionProvider` from the providers list.** onnxruntime
always appends CPU implicitly for unsupported ops, and the preprocessor/resampler genuinely needs
it. Removing it does not do what it looks like it does.

## Troubleshooting

| Symptom | Look at |
|---|---|
| Shortcut does nothing | Is the daemon alive? `pgrep -af dictate_daemon.py`, then `~/.local/share/dictate/daemon.log`. |
| Records, but no text appears | The ydotool wrapper — is `ydotoold` running, does `wl-copy` work, does the window take Ctrl+V? |
| Paste inserts your *old* clipboard | The restore delay is too short for that app: raise `YDOTOOL_CLIP_RESTORE_DELAY`. |
| Suddenly ~5x slower | It's on CPU. `nvidia-smi` (above); expect a FATAL in the log now. |
| `Nothing recognized.` | Empty/silent capture — check the PipeWire input source. |
| VRAM needed elsewhere | `dictate-stop`; `dictate-restart` brings it back. |

The daemon logs to `~/.local/share/dictate/daemon.log` (not in this repo).
