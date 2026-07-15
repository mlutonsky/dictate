#!/bin/sh
# ydotool wrapper v6 - Increased delays for reliability

LOG_FILE="/tmp/hyprvoice-debug.log"

# Clean the log for the new test
echo "" > "$LOG_FILE"

subcommand="$1"
shift

if [ "$subcommand" = "type" ]; then
  # Remove the leading "--" if it exists
  if [ "$1" = "--" ]; then
    shift
  fi
  
  text_to_paste="$@"
  
  # Copy cleaned text to the clipboard
  printf "%s" "$text_to_paste" | wl-copy

  # Increased wait time for clipboard to be ready
  sleep 0.25

  # Use the real ydotool with a generous 100ms delay
  /usr/bin/ydotool key --key-delay 100 29:1 47:1 47:0 29:0
else
  # For any other command, just pass it through
  /usr/bin/ydotool "$subcommand" "$@"
fi
