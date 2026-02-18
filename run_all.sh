#!/usr/bin/env bash
set -euo pipefail

cd /home/sr2/kjh9503/homebench_graph
rm -rf output/*
rm -rf ~/.local/share/homebench_graph
# QNUM=11 crewai run
# QNUM=50 crewai run
# QNUM=62 crewai run
for qnum in $(seq 0 99); do
  echo "Running QNUM=${qnum}"
  QNUM="${qnum}" crewai run
done