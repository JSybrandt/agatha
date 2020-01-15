#include<string>
#include<sstream>
#include<filesystem>
#include <sqlite3.h>
namespace fs = std::filesystem;

void add_index_to_sqlite(
    fs::path sqlite_path,
    std::string index_name,
    std::string table_name,
    std::string column_name
){
  // Now we need to index the added entity column.
  // This is faster than creating one to begin with.
  std::stringstream index_stmt;
  index_stmt << "CREATE INDEX IF NOT EXISTS "
             << index_name
             << " ON "
             << table_name
             << "("
             << column_name
             << ");";

  // Creating index on entity column
  sqlite3* db_conn;
  int exit_code = sqlite3_open(sqlite_path.c_str(), &db_conn);
  if(exit_code){
    std::cerr << "Failed to open " << sqlite_path << std::endl;
  }
  char* err_msg;
  exit_code = sqlite3_exec(
      db_conn,
      index_stmt.str().c_str(),
      0,
      0,
      &err_msg
  );
  if(exit_code){
    std::cerr << "Failed to create index:" << err_msg << std::endl;
  }
  sqlite3_close(db_conn);
}
