#include <argparse.hpp>
#include <iostream>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <sqlite_orm/sqlite_orm.h>
#include <fstream>
#include <sstream>
#include <cppitertools/enumerate.hpp>
#include <sqlite3.h>
#include "glob.h"
#include "add_index_to_sqlite.h"

using std::stringstream;
using std::cout;
using std::endl;
using std::runtime_error;
using std::vector;
using std::list;
using std::string;
using std::fstream;
using std::ios;
using std::tuple;
using json = nlohmann::json;
namespace fs = std::filesystem;
namespace sql = sqlite_orm;

// This contains enough information to recover an embedding from an h5 file.
struct EntityEmbeddingLocation {
  string entity;
  string entity_type;
  size_t partition_idx;
  size_t row_idx;
};

// Parses the json file name (expected to be entity_names_{type}_{part}.json)
tuple<char, size_t> parse_json_filename(const fs::path json_path){
  assert(json_path.extension() == ".json");
  stringstream parser(json_path.stem());
  string entity, names, type;
  size_t part;
  try{
    getline(parser, entity, '_');
    assert(entity == "entity");
    getline(parser, names, '_');
    assert(names == "names");
    getline(parser, type, '_');
    assert(type.size() == 1);
    parser >> part;
  } catch(...){
    throw runtime_error("Invalid file name: " + string(json_path));
  }
  return {type[0], part};
}

list<EntityEmbeddingLocation> file_to_emb_locs(const fs::path& json_path){
  auto [type, part] = parse_json_filename(json_path);
  json names;
  fstream json_file(json_path, ios::in);
  json_file >> names;
  json_file.close();
  list<EntityEmbeddingLocation> result;
  for(auto [row, name]: iter::enumerate(names)){
    result.push_back({name, string(1, type), part, row});
  }
  return result;
}

int main(int argc, char **argv){
  argparse::ArgumentParser parser("ptbg_ents_to_sqlite");
  parser.add_argument("-i", "--ent-dir")
        .help("Location containing entity_name_*.json files.")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-o", "--sqlite")
        .help("The location to write sqlite db")
        .action([](const std::string& s){ return fs::path(s); });
  parser.add_argument("-n", "--table-name")
        .help("The name of the table within the sqlite3 database.")
        .default_value(string("embedding_locations"));
  try {
    parser.parse_args(argc, argv);
  }
  catch (const runtime_error& err) {
    cout << err.what() << endl;
    cout << parser;
    return 1;
  }

  fs::path ent_dir_path = parser.get<fs::path>("--ent-dir");
  fs::path sqlite_path = parser.get<fs::path>("--sqlite");
  string table_name = parser.get<string>("--table-name");

  assert(fs::is_directory(ent_dir_path));
  assert(!fs::exists(sqlite_path));

  vector<fs::path> all_json_files = glob_ext(ent_dir_path, ".json");
  assert(all_json_files.size() > 0);

  auto storage = sql::make_storage(
      sqlite_path,
      sql::make_table(
        table_name,
        sql::make_column(
          "entity",
          &EntityEmbeddingLocation::entity
        ),
        sql::make_column(
          "entity_type",
          &EntityEmbeddingLocation::entity_type
        ),
        sql::make_column(
          "partition_idx",
          &EntityEmbeddingLocation::partition_idx
        ),
        sql::make_column(
          "row_idx",
          &EntityEmbeddingLocation::row_idx
        )
      )
  );
  storage.sync_schema();


  cout << "Loading embedding locations" << endl;
  list<EntityEmbeddingLocation> all_embedding_locations;
  #pragma omp parallel for schedule(dynamic)
  for(size_t path_idx = 0; path_idx < all_json_files.size(); ++path_idx){
    auto tmp = file_to_emb_locs(all_json_files[path_idx]);
    #pragma omp critical
    {
      all_embedding_locations.splice(all_embedding_locations.end(), tmp);
    }
  }

  cout << "Writing DB" << endl;
  storage.transaction([&]{
      for(auto& e: all_embedding_locations){
        storage.insert(e);
      }
      return true;
  });

  cout << "Creating index" << endl;
  add_index_to_sqlite(
      sqlite_path,
      "entity_index",
      table_name,
      "entity"
  );
  return 0;
}
