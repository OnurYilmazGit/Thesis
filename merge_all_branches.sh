#!/bin/bash

# Check if the current branch is 'merged-branch'
current_branch=$(git branch --show-current)
if [ "$current_branch" != "merged-branch" ]; then
  echo "Please switch to the 'merged-branch' before running this script."
  exit 1
fi

# Fetch all remote branches
git fetch --all

# List all remote branches (excluding HEAD pointer)
branches=$(git branch -r | grep -v 'HEAD' | sed 's/origin\///')

# Merge each remote branch into 'merged-branch'
for branch in $branches; do
  echo "Merging branch: origin/$branch"
  git merge origin/$branch --strategy-option theirs -m "Merged branch origin/$branch"
  
  if [ $? -ne 0 ]; then
    echo "Merge conflict occurred while merging origin/$branch."
    echo "Please resolve the conflicts manually and continue the process."
    exit 1
  fi
done

echo "All branches successfully merged into 'merged-branch'."

