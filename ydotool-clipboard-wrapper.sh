#!/bin/sh
# ydotool wrapper — installed as /usr/local/bin/ydotool, shadowing /usr/bin/ydotool in PATH.
#
# Intercepts `ydotool type` and pastes via the clipboard instead of typing keystrokes.
# This is not a preference — on GNOME/Wayland it's the only thing that works:
#   * `ydotool type` emits raw US keycodes, so it cannot produce Czech diacritics at all
#     and mangles even ASCII while the cz+qwerty layout is active.
#   * `wtype` needs the virtual-keyboard protocol, which Mutter does not implement.
#
# The clipboard is saved before the paste and restored afterwards, so dictating no longer
# destroys whatever you had copied. Everything except `type` is passed straight through.

if [ "$1" != "type" ]; then
    exec /usr/bin/ydotool "$@"
fi
shift
[ "$1" = "--" ] && shift
text="$*"

# How long to wait after Ctrl+V before restoring the old clipboard. The target app requests
# the data within milliseconds of the keypress, so this is a wide margin — but if it were
# ever too short, the app would read the *restored* content and paste the wrong text.
# Restoring cannot be synchronised properly: wl-copy --paste-once would report exactly when
# the paste happened, but GNOME's clipboard manager reads the selection itself and consumes
# that single serve immediately, so the signal is worthless here.
restore_delay="${YDOTOOL_CLIP_RESTORE_DELAY:-0.5}"

# --- save the current clipboard -------------------------------------------------
# One MIME type is restored, not the full set an app may offer: preferring plain UTF-8 text
# covers the common case, and image/* survives too. Multi-type offers (e.g. PhpStorm's Java
# cookie plus a dozen text encodings) cannot be reproduced faithfully by wl-copy.
saved="$(mktemp)"
saved_type=""
types="$(wl-paste --list-types 2>/dev/null)"
if [ -n "$types" ]; then
    saved_type="$(printf '%s\n' "$types" | grep -ixm1 'text/plain;charset=utf-8')"
    [ -z "$saved_type" ] && saved_type="$(printf '%s\n' "$types" | grep -ixm1 'text/plain')"
    [ -z "$saved_type" ] && saved_type="$(printf '%s\n' "$types" | head -n1)"
    wl-paste --no-newline --type "$saved_type" > "$saved" 2>/dev/null || saved_type=""
fi

# --- paste the new text ---------------------------------------------------------
printf '%s' "$text" | wl-copy
sleep 0.25                                                 # let wl-copy own the selection
/usr/bin/ydotool key --key-delay 100 29:1 47:1 47:0 29:0   # Ctrl+V
rc=$?

# --- put the old clipboard back -------------------------------------------------
sleep "$restore_delay"
if [ -n "$saved_type" ] && [ -s "$saved" ]; then
    wl-copy --type "$saved_type" < "$saved"
else
    wl-copy --clear                                        # it was empty before; leave it empty
fi
rm -f "$saved"

exit $rc
