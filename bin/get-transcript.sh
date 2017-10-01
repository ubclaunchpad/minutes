#!/usr/bin/env bash
#
# Downloads audio and transcript of a YouTube video to the current directory.
# Requires `youtube-dl` and `ffmpeg`.

TRANSCRIPT_BASE="https://www.youtube.com/api/timedtext?lang=en&v="

if [ -z "$1" ]; then
  echo "Usage: ./get-transcript.sh <VIDEO_ID>"
  exit 2
fi

# Attempt to download XML transcript
if curl --silent --fail --output "$1.xml" "${TRANSCRIPT_BASE}${1}"; then
  echo "Downloaded transcript to $1.xml."
else
  echo "Unable to download transcript."
  exit 1
fi

youtube-dl --all-subs --extract-audio --audio-format="wav" --output="$1/$1.%(ext)s" "$1"
