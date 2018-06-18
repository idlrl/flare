#!/usr/bin/env bash

function init() {
    RED='\033[0;31m'
    BLUE='\033[0;34m'
    BOLD='\033[1m'
    NONE='\033[0m'

    REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}")/../" && pwd )"
}

function print_usage() {
    echo -e "\n${RED}Usage${NONE}:
    ${BOLD}$0${NONE} [OPTION]"

    echo -e "\n${RED}Options${NONE}:
    ${BLUE}test${NONE}: run all unit tests
    ${BLUE}check_style${NONE}: run code style check
    "
}

function abort(){
    echo "Your change doesn't follow the code style." 1>&2
    echo "Please use pre-commit to check what is wrong." 1>&2
    exit 1
}

function check_style() {
    trap 'abort' 0
    set -e

    pip install pre-commit

    export PATH=/usr/bin:$PATH
    pre-commit install

    if ! pre-commit run -a ; then
        git diff
        exit 1
    fi

    trap : 0
}

function run_test() {
    mkdir -p ${REPO_ROOT}/build
    cd ${REPO_ROOT}/build
    cmake ..
    cat <<EOF
    ========================================
    Running unit tests ...
    ========================================
EOF
    ctest --output-on-failure
}

function main() {
    set -e
    local CMD=$1
    init
    case $CMD in
        check_style)
          check_style
          ;;
        test)
          run_test
          ;;
        *)
          print_usage
          exit 0
          ;;
    esac
}

main $@
