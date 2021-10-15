extern "C" {
#include "mz.h"
#include "mz_os.h"
#include "mz_strm.h"
#include "mz_strm_buf.h"
#include "mz_strm_os.h"
#include "mz_strm_split.h"
#include "mz_zip.h"
#include "mz_zip_rw.h"
}

#include <iostream>

int main(int argc, char *argv[]) {
  void *file_stream = NULL;
  void *zip_reader = NULL;
  const char *path = "/home/zack/test.zip";

  mz_zip_reader_create(&zip_reader);
  mz_stream_os_create(&file_stream);

  auto err =
      mz_stream_os_open(file_stream, path, MZ_OPEN_MODE_READ); // 打开文件
  if (err == MZ_OK) {
    err = mz_zip_reader_open(zip_reader, file_stream); //从流中读取zip文件
    if (err == MZ_OK) {
      printf("Zip reader was opened %s\n", path);
      if (mz_zip_reader_goto_first_entry(zip_reader) == MZ_OK) {
        do {
          mz_zip_file *file_info = NULL;
          if (mz_zip_reader_entry_get_info(zip_reader, &file_info) != MZ_OK) {
            printf("Unable to get zip entry info\n");
            break;
          }
          printf("Zip entry %s\n", file_info->filename);
        } while (mz_zip_reader_goto_next_entry(zip_reader) == MZ_OK);
      }
      const char *search_filename = "test.txt";
      if (mz_zip_reader_locate_entry(zip_reader, search_filename, 1) == MZ_OK)
        printf("Found %s\n", search_filename);
      else
        printf("Could not find %s\n", search_filename);
      mz_zip_reader_close(zip_reader);
    }
  }

  mz_stream_os_delete(&file_stream);
  mz_zip_reader_delete(&zip_reader);
}
