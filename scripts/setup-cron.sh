#!/bin/bash

# Setup weekly retraining cron job
# Run this script to install the cron job for automated weekly retraining

PROJECT_DIR=$(pwd)
CRON_JOB="0 2 * * 0 cd $PROJECT_DIR && python retrain_pipeline.py >> $PROJECT_DIR/cron.log 2>&1"

echo "Setting up weekly retraining cron job..."
echo "Project directory: $PROJECT_DIR"
echo "Cron job: $CRON_JOB"

# Add to crontab if not already present
(crontab -l 2>/dev/null | grep -v "retrain_pipeline.py"; echo "$CRON_JOB") | crontab -

echo "Cron job installed successfully!"
echo "The model will retrain every Sunday at 2:00 AM"
echo "Logs will be written to: $PROJECT_DIR/cron.log"

# Create log file
touch "$PROJECT_DIR/cron.log"
echo "Created log file: $PROJECT_DIR/cron.log"

echo ""
echo "To verify the cron job was installed:"
echo "  crontab -l"
echo ""
echo "To remove the cron job:"
echo "  crontab -e  # then delete the retrain_pipeline.py line"
