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


struct BowEntry {
  std::string id;
  std::string json_encoded_bow;
  BowEntry(std::string i, std::string j):id(i), json_encoded_bow(j){};
};

std::list<BowEntry> parse_json_file(const fs::path& json_path){
  std::list<BowEntry> res;
  std::fstream json_file(json_path, std::ios::in);
  std::string line;
  while(getline(json_file, line)){
    json id_bow = json::parse(line);
    res.emplace_back(id_bow["id"], id_bow["bow"].dump());
  }
  json_file.close();
  return res;
}

int main(int argc, char** argv){
  argparse::ArgumentParser parser("bow_json_to_sqlite");
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

  std::cout << "Loading all bow" << std::endl;
  int num_finished = 0;
  std::list<BowEntry> bag_entries;
  #pragma omp parallel for schedule(dynamic)
  for(size_t i = 0; i < all_json_files.size(); ++i){
    std::list<BowEntry> local_bags = parse_json_file(all_json_files[i]);
    #pragma omp critical
    {
      bag_entries.splice(bag_entries.end(), local_bags);
      ++num_finished;
      std::cout << num_finished << "/" << all_json_files.size() << std::endl;
    }
  }

  std::cout << "Creating Database" << std::endl;
  auto storage = sql::make_storage(
      sqlite_path,
      sql::make_table(
        "sentences",
        sql::make_column("id", &BowEntry::id),
        sql::make_column("bow", &BowEntry::json_encoded_bow)
      )
  );
  storage.sync_schema();

  std::cout << "Writing DB" << std::endl;
  storage.transaction([&]{
      for(const auto& e: bag_entries){
        storage.insert(e);
      }
      return true;
  });

  std::cout << "Adding id index to db" << std::endl;
  add_index_to_sqlite(
      sqlite_path,
      "id_index",
      "sentences",
      "id"
  );
  return 0;
}
