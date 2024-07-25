#!/bin/bash

# Function to show script usage
usage() {
    echo "Usage: $0 --path <directory-path> --variables <var1,var2,...> --runs <number-of-runs>"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --path) DIR_PATH="$2"; shift ;;  # Directory path
        --variables)
            # Split the string into an array and trim spaces from each element
            IFS=',' read -r -a temp_variables <<< "$2"
            VARIABLES=()
            for var in "${temp_variables[@]}"; do
                # Trim leading and trailing whitespace
                trimmed_var=$(echo "$var" | xargs)
                VARIABLES+=("$trimmed_var")
            done
            shift ;;
        --runs) RUNS="$2"; shift ;;  # Number of runs
        *) usage ;;  # Display usage if arguments are incorrect
    esac
    shift
done

# Check if all required arguments are provided
if [[ -z "$DIR_PATH" || -z "${VARIABLES[*]}" || -z "$RUNS" ]]; then
    echo "Error: Missing required arguments."
    usage
fi

# Check if the specified directory exists
if [[ ! -d "$DIR_PATH" ]]; then
    echo "Error: Directory $DIR_PATH does not exist."
    exit 1
fi

# Display the input parameters for verification
echo "Checking directory: $DIR_PATH"
echo "Variables: ${VARIABLES[*]}"
echo "Number of runs: $RUNS"

# Function to check for missing .sca files
check_missing_files() {
    local variable=$1  # Variable name
    local run_count=$2  # Number of runs
    local var_index=$3  # Index of the variable in the list
    local missing_runs=""  # Initialize empty string to store missing runs

    for (( run=0; run<run_count; run++ )); do
        # Construct the expected file name pattern
        local expected_file="$DIR_PATH/*=$variable-#$run.sca"
        # Calculate the global run number
        local global_run_number=$((var_index * run_count + run))

        # Check if the .sca file exists
        if ! compgen -G "$expected_file" > /dev/null; then
            # Print missing file information to stderr
            echo "Missing .sca file: $expected_file (Global Run: $global_run_number)" >&2
            # Append the missing run information
            missing_runs+="${variable} $run $global_run_number,"
        fi
    done

    echo $missing_runs  # Return the list of missing runs
}

# Check for missing files across all variables
all_missing_runs=""
for i in "${!VARIABLES[@]}"; do
    variable=$(echo ${VARIABLES[$i]} | xargs)  # Trim whitespace from variable name
    # Get missing runs for the current variable
    missing=$(check_missing_files "$variable" "$RUNS" "$i")
    if [[ -n "$missing" ]]; then
        # Aggregate missing runs from all variables
        all_missing_runs+="$missing"
    fi
done

# Format and output the missing runs
if [[ -n "$all_missing_runs" ]]; then
    # Remove the trailing comma from the aggregated string
    all_missing_runs=${all_missing_runs%,}

    # Format the output with commas and without spaces
    formatted_output="Missing runs:"
    IFS=',' read -ra RUNS_ARRAY <<< "$all_missing_runs"
    for run_info in "${RUNS_ARRAY[@]}"; do
        formatted_output+="${run_info##* },"
    done
    formatted_output=${formatted_output%,}  # Remove the final comma
    echo "$formatted_output"

    # Output in tabular format
    echo "Tabular format of missing runs:"
    echo -e "Variable\tLocal Run\tGlobal Run"
    for run_info in "${RUNS_ARRAY[@]}"; do
        IFS=' ' read -ra INFO <<< "$run_info"
        # Print each row of the table
        echo -e "${INFO[0]}\t\t${INFO[1]}\t\t${INFO[2]}"
    done
else
    echo "No missing runs found."
fi
