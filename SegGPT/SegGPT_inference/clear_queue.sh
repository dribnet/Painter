#!/bin/bash

# continue on error (files will be moved to working)
# set -ex

# NOTE : Quote it else use array to avoid problems #
FILES="/dropbox/segments/01_queue/*"
for ORIG_PATH in $FILES
do
  FPATH="${ORIG_PATH// /_}"

  if [ "$ORIG_PATH" != "$FPATH" ]; then
    echo "File rename: $FPATH"
      mv "$ORIG_PATH" "$FPATH";
  fi

  BASENAME=$(basename $FPATH)
  NEWNAME="/dropbox/segments/02_working/$BASENAME"
  echo "Processing $BASENAME"
  mv $FPATH $NEWNAME
  # take action on each file. $f store current file name
  ./run_queue.sh "$NEWNAME"
done
