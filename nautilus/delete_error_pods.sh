#!/bin/bash

# Get all jobs with their names and statuses, filtering for those starting with 'alopez'
jobs=$(kubectl get jobs -o jsonpath="{range .items[*]}{.metadata.name} {.status.conditions[*].type}{'\n'}{end}" | grep '^alopez')

if [ -z "$jobs" ]; then
  echo "No jobs found that start with 'alopez'."
else
  echo "Found jobs starting with 'alopez':"
  
  # Filter jobs by their statuses
  non_matching_jobs=()
  while IFS= read -r line; do
    job_name=$(echo "$line" | awk '{print $1}')
    job_status=$(echo "$line" | awk '{print $3}')
    
    # Only delete jobs that are not "Running", "Pending", or "ContainerCreating"
    if [[ "$job_status" != "Running" && "$job_status" != "Pending" && "$job_status" != "ContainerCreating" ]]; then
      non_matching_jobs+=("$job_name")
    fi
  done <<< "$jobs"
  
  if [ ${#non_matching_jobs[@]} -eq 0 ]; then
    echo "No 'alopez' jobs found with statuses other than 'Running', 'Pending', or 'ContainerCreating'."
  else
    echo "Deleting 'alopez' jobs with statuses other than 'Running', 'Pending', or 'ContainerCreating'..."
    for job in "${non_matching_jobs[@]}"; do
      echo "Deleting job: $job"
      kubectl delete job "$job"
    done
    echo "All non-matching 'alopez' jobs have been deleted."
  fi
fi
