#!/usr/bin/env bashc

while getopts c:i: flag
do
    # shellcheck disable=SC2220
    case "${flag}" in
        c) config_mode=${OPTARG};;
        i) id_exp=${OPTARG};;
    esac
done
echo "config_mode: $config_mode";
echo "id_exp: $id_exp";


