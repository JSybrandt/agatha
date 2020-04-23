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
#include <fstream>
#include <list>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <argparse.hpp>
#include <cppitertools/zip.hpp>
#include <nlohmann/json.hpp>
#include <sqlite_orm/sqlite_orm.h>

#include "add_index_to_sqlite.h"
#include "glob.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace sql = sqlite_orm;

struct TableEntry {
  TableEntry(
      const std::string& k,
      const std::string& v
  ): key(k), value(v) {}
  std::string key;
  std::string value;
};

std::string get_value_string(const json& val){
  if (val.is_object() || val.is_array()){
    return val.dump();
  } else {
    return val.get<std::string>();
  }
}

std::list<TableEntry> parse_json(const fs::path& json_path){
  std::list<TableEntry> res;
  std::fstream json_file(json_path, std::ios::in);
  std::string line;
  while (getline(json_file, line)){
    json key_value_entry = json::parse(line);
    assert(key_value_entry.find("key") != key_value_entry.end());
    assert(key_value_entry.find("value") != key_value_entry.end());
    assert(key_value_entry["key"].is_string());
    res.emplace_back(
        key_value_entry["key"].get<std::string>(),
        get_value_string(key_value_entry["value"])
    );
  }
  json_file.close();
  return res;
}

void merge_duplicate_keys(std::list<TableEntry>& table_entries){
  std::unordered_map<std::string, std::unordered_set<std::string>> key2values;
  for (const auto& entry : table_entries){
    key2values[entry.key].insert(entry.value);
  }
  table_entries.clear();
  for(const auto& [key, value_set] : key2values){
    table_entries.emplace_back(
      key,
      json(value_set).dump()
    );
  }
}

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
  std::list<TableEntry> table_entries;
  #pragma omp parallel
  {
    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < all_json_files.size(); ++i){
      std::list<TableEntry> tmp = parse_json(all_json_files[i]);
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
        sql::make_column("key", &TableEntry::key),
        sql::make_column("value", &TableEntry::value)
      )
  );
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
