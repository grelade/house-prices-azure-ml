#!/bin/sh

# Set Kaggle configuration directory
export KAGGLE_CONFIG_DIR="."

# Initialize variables with default values
RESULTS_DIR=""
SUBMISSION_MESSAGE="aml_submission_msg"

# Function to display help message
show_help() {
    echo "Usage: kaggle_upload.sh --results_dir RESULTS_DIR --submission_message SUBMISSION_MESSAGE"
    echo
    echo "Options:"
    echo "  --results_dir RESULTS_DIR "
    echo "  --submission_message SUBMISSION_MESSAGE default = 'aml_submission_msg'"
    echo "  -h, --help         Display this help message."
    echo
}

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --results_dir)
            RESULTS_DIR="$2"
            shift # past argument
            shift # past value
            ;;
        --submission_message)
            SUBMISSION_MESSAGE="$2"
            shift # past argument
            shift # past value
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Invalid option $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if required arguments are provided
if [ -z "$RESULTS_DIR" ]; then
    echo "Error: --results_dir is required."
    show_help
    exit 1
fi

# Echo the parameters (for debugging purposes)
echo "Submitting file: $RESULTS_DIR/sample_submission.csv"
echo "Submission message: $SUBMISSION_MESSAGE"

# Kaggle submission command
kaggle competitions submit -f "$RESULTS_DIR/sample_submission.csv" -m "$SUBMISSION_MESSAGE" house-prices-advanced-regression-techniques

