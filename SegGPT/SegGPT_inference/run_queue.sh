#!/bin/bash

# stop on error
set -ex

FPATH="$1"

BASEFILE="$(basename $FPATH)"
BASENAME="${BASEFILE%.*}"
TODAY=$(date +'%Y_%m_%d')
WORKDIR="processing/$BASENAME"
DATEDIR="/dropbox/segments/03_complete/"$TODAY
OUTDIR="$DATEDIR"/"$BASENAME"

echo "examining $BASENAME"

cp -r $FPATH $WORKDIR

python -u seggpt_inference_tom2.py --in_dir $WORKDIR | tee "$WORKDIR"/result.txt

mkdir -p $DATEDIR
mv $FPATH $OUTDIR
cp "$WORKDIR"/mask_new*.png "$OUTDIR"/.
cp "$WORKDIR"/result.txt "$OUTDIR"/result.txt

#echo "future: rm -Rf $WORKDIR"
rm -Rf $WORKDIR

echo "Saved results to $OUTDIR"
