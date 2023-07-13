#!/bin/bash

rc=0;
for pre in asl ds00 eeg meg ieeg; do
    for i in $(find bids-examples -maxdepth 1 -type d -name "${pre}*"); do
        if [ "$i" == "bids-examples" ]; then
            continue
        fi
        echo
        echo "running on dataset" $i
        CMD="pybids_reports ${i%%/} ${PWD} --verbosity 1"
        echo "$CMD"
        $CMD || rc=$?
    done
done
exit $rc;
