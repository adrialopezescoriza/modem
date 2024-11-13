#!/bin/bash
jobs=$(kubectl get jobs -o jsonpath="{.items[*].metadata.name}" | tr ' ' '\n' | grep '^alopez')

if [ -z "$jobs" ]; then
  echo "No jobs found that start with 'alopez'."
else
  echo "Found jobs:"
  echo "$jobs"
  echo "Deleting all jobs..."
  for job in $jobs; do
    kubectl delete job $job
  done
  echo "All matching jobs have been deleted."
fi
