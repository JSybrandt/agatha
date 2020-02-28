#include <unordered_map>
#include <list>
#include <unordered_set>
#include <argparse.hpp>
#include <nlohmann/json.hpp>
#include <sqlite_orm/sqlite_orm.h>
#include <sqlite3.h>
#include <filesystem>
#include <fstream>
#include <tuple>
#include <sstream>
#include <cppitertools/zip.hpp>
#include "glob.h"
#include "add_index_to_sqlite.h"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace sql = sqlite_orm;

using Values = std::list<std::string>;
using LoopupTable = std::unordered_map<std::string, Values>;

struct TableEntry {
  std::string key;
  std::string value;
};

void merge_tables(LoopupTable& base, LoopupTable& addition){
  for(auto& [key, added_values] : addition){
    Values& base_neighbors = base[key];
    base_neighbors.splice(base_neighbors.end(), added_values);
  }
}

LoopupTable parse_json(const fs::path& json_path){
  LoopupTable res;
  std::fstream json_file(json_path, std::ios::in);
  std::string line;
  while(getline(json_file, line)){
    json kv_pair = json::parse(line);
    res[kv_pair[0]].push_back(kv_pair[1]);
  }
  json_file.close();
  return res;
}

int main(int argc, char** argv){
  argparse::ArgumentParser parser("create_lookup_table");
  parser.add_argument("-i", "--json-dir")
        .help("Location containing json files.")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-o", "--sqlite")
        .help("The location to write sqlite db")
        .action([](const std::string& s){ return fs::path(s); });
  try {
    parser.parse_args(argc, argv);
  }
  catch (const std::runtime_error& err) {
    std::cout << err.what() << std::endl;
    std::cout << parser;
    return 1;
  }

  fs::path json_dir_path = parser.get<fs::path>("--json-dir");
  fs::path sqlite_path = parser.get<fs::path>("--sqlite");

  assert(fs::is_directory(json_dir_path));
  assert(!fs::exists(sqlite_path));

  std::vector<fs::path> all_json_files = glob_ext(json_dir_path, ".json");
  assert(all_json_files.size() > 0);

  int num_finished = 0;
  LoopupTable table;
  #pragma omp parallel
  {
    LoopupTable local_table;
    #pragma omp for schedule(dynamic)
    for(size_t i = 0; i < all_json_files.size(); ++i){
      LoopupTable tmp = parse_json(all_json_files[i]);
      merge_tables(local_table, tmp);
      #pragma omp critical
      {
        ++num_finished;
        std::cout << num_finished << "/" << all_json_files.size() << std::endl;
      }
    }
    #pragma omp critical
    {
      merge_tables(table, local_table);
    }
  }

  std::vector<std::string> keys;
  keys.reserve(table.size());
  for(const auto& [key, values] : table){
    keys.push_back(key);
  }

  std::vector<std::string> values(keys.size());
  #pragma omp parallel for schedule(dynamic)
  for(size_t i=0; i<keys.size(); ++i){
    auto& value_list = table[keys[i]];
    std::unordered_set<std::string> value_set(
        value_list.begin(), value_list.end()
    );
    values[i] = json(value_set).dump();
    table[keys[i]].clear(); // drop list, keep memory reasonable
  }

  std::cout << "Creating Database" << std::endl;
  auto storage = sql::make_storage(
      sqlite_path,
      sql::make_table(
        "lookup_table",
        sql::make_column("key", &TableEntry::key),
        sql::make_column("value", &TableEntry::value)
      )
  );
  storage.sync_schema();

  std::cout << "Writing DB" << std::endl;
  storage.transaction([&]{
      // Both strings
      for(const auto&& [key, value]: iter::zip(keys, values)){
        storage.insert(TableEntry{key, value});
      }
      return true;
  });

  std::cout << "Adding key index to db" << std::endl;
  add_index_to_sqlite(
      sqlite_path,
      "key_index",
      "lookup_table",
      "key"
  );
  return 0;
}
