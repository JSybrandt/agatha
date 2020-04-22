#/bin/bash

function run_test(){
  d=$1
  flg=$2

  echo "=== $d ==="

  rm -f $d/db.sqlite3

  ../create_lookup_table \
    -i $d \
    -o $d/db.sqlite3 \
    $flg

  echo "=== INPUT DATA==="
  cat $d/*.json

  echo "=== ACTUAL  ==="
  sqlite3 $d/db.sqlite3 "SELECT * FROM lookup_table;"

  # Cleanup
  rm -f $d/db.sqlite3
}

run_test complex_no_dup
run_test simple_no_dup
run_test simple_has_dup -m
