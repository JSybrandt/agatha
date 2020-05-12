/*
 * Create Lookup Table
 *
 * Accepts a directory of JSON files, creates a sqlite3 table.
 *
 * Expected JSON schema:
 *
 * {
 *  "key": "key_string"
 *  "value" "value_string" OR {<Value Object>}
 * }
 *
 * Produces an sqlite3 database with:
 *
 * CREATE TABLE IF NOT EXISTS 'lookup_table' (
 *   'key' TEXT NOT NULL,
 *   'value' TEXT NOT NULL
 * );
 * CREATE INDEX key_index ON lookup_table(key);
 *
 *
 */
#include <sqlite3.h>

#include <filesystem>
#include <list>
#include <vector>

#include <argparse.hpp>
#include <nlohmann/json.hpp>
#include <sqlite_orm/sqlite_orm.h>

#include "add_index_to_sqlite.h"
#include "glob.h"
#include "parse_kv_json.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace sql = sqlite_orm;


int main(int argc, char** argv){
  argparse::ArgumentParser parser("create_lookup_table");
  parser.add_argument("-i", "--json-dir")
        .help("Location containing json files.")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-o", "--sqlite")
        .help("The location to write sqlite db")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-m", "--merge-duplicates")
        .help("If set, values of duplicate keys will be combined as sets.")
        .default_value(false)
        .implicit_value(true);
  parser.add_argument("-v", "--verbose")
        .help("If set, writes details to stdout.")
        .default_value(false)
        .implicit_value(true);
  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cerr<< err.what() << std::endl;
    std::cerr<< parser;
    return 1;
  }

  /*  ARGUMENTS PARSED  */

  fs::path json_dir_path = parser.get<fs::path>("--json-dir");
  fs::path sqlite_path = parser.get<fs::path>("--sqlite");
  bool merge_duplicates = parser.get<bool>("--merge-duplicates");
  bool verbose = parser.get<bool>("--verbose");

  assert(fs::is_directory(json_dir_path));
  assert(!fs::exists(sqlite_path));

  std::vector<fs::path> all_json_files = glob_ext(json_dir_path, ".json");
  assert(all_json_files.size() > 0);

  // List used because splice is O(1) and vector merge is O(n)
  size_t num_finished = 0;
  std::list<KVPair> table_entries;
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < all_json_files.size(); ++i){
      std::list<KVPair> tmp = parse_kv_json(all_json_files[i]);
      #pragma omp critical
      {
        table_entries.splice(table_entries.end(), tmp);
        ++num_finished;
        if (verbose && num_finished % 10)
          std::cout << num_finished
                    << "/"
                    << all_json_files.size()
                    << std::endl;
      }
    }
  }

  if (merge_duplicates){
    if (verbose) std::cout << "Merging Duplicate Keys" << std::endl;
    merge_duplicate_keys(table_entries);
  }


  if (verbose) std::cout << "Creating Database" << std::endl;
  auto storage = sql::make_storage(
      sqlite_path,
      sql::make_table(
        "lookup_table",
        sql::make_column("key", &KVPair::key),
        sql::make_column("value", &KVPair::value)
      )
  );
  storage.pragma.journal_mode(sqlite_orm::journal_mode::OFF);
  storage.sync_schema();

  if (verbose) std::cout << "Writing DB" << std::endl;
  storage.transaction([&]{
      // Both strings
      for(const auto& table_entry : table_entries){
        storage.insert(table_entry);
      }
      return true;
  });

  if (verbose) std::cout << "Adding key index to db" << std::endl;
  add_index_to_sqlite(
      sqlite_path,
      "key_index",
      "lookup_table",
      "key"
  );
  return 0;
}
