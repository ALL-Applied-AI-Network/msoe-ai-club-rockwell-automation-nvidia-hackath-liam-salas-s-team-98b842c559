#!/usr/bin/env bash
# Prepare one fresh Brev L40S host for queue dispatch: Isaac 6.2 runtime plus
# the frozen source, plan, qualification receipt and twin files, at the same
# remote paths the agent's launchers use. Run from the GB10.
#
# usage: bootstrap_host.sh HOST SOURCE_TAR SOURCE_NAME PLAN_JSON RECEIPT_JSON RECEIPT_REMOTE_PATH SETUP_SH TWIN_TSV
set -euo pipefail

host=$1 source_tar=$2 source_name=$3 plan=$4 receipt=$5 receipt_remote=$6 setup=$7 twin_tsv=$8
base=/home/ubuntu/final-recipe8

retry() { for i in 1 2 3 4; do "$@" && return 0; sleep $((15 * i)); done; return 1; }

retry brev exec "$host" "mkdir -p $base/code/$source_name $base/payloads $base/runs $(dirname "$receipt_remote") $base/twin/geometry/textures"
retry brev copy "$setup" "$host:/home/ubuntu/setup-isaac62.sh"
retry brev copy "$source_tar" "$host:/home/ubuntu/source.tar.gz"
retry brev copy "$plan" "$host:$base/payloads/$(basename "$plan")"
retry brev copy "$receipt" "$host:$receipt_remote"
tail -n +2 "$twin_tsv" | while IFS=$'\t' read -r _sha local remote; do
  retry brev copy "$local" "$host:$remote"
done

# Setup runs in the foreground here; it took about 3 minutes on the agent's hosts.
retry brev exec "$host" "tar -xzf /home/ubuntu/source.tar.gz -C $base/code/$source_name && bash /home/ubuntu/setup-isaac62.sh > /home/ubuntu/setup.log 2>&1; tail -3 /home/ubuntu/setup.log"

# Hash check of everything the supervisor binds to.
brev exec "$host" "sha256sum /home/ubuntu/source.tar.gz $base/payloads/$(basename "$plan") $receipt_remote $(tail -n +2 "$twin_tsv" | cut -f3 | tr '\n' ' ')"
