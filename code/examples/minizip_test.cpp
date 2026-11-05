/*
 * @Author: zack 
 * @Date: 2021-10-17 13:34:08 
 * @Last Modified by: zack
 * @Last Modified time: 2021-10-17 15:44:26
 */
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
  const char *path = "test.zip";

  mz_zip_reader_create(&zip_reader);
  mz_stream_os_create(&file_stream);

  auto err =
      mz_stream_os_open(file_stream, path, MZ_OPEN_MODE_READ); // 打开文件
  if (err == MZ_OK) {
    err = mz_zip_reader_open(zip_reader, file_stream); //从流中读取zip文件
    if (err == MZ_OK) {
      printf("Zip reader was opened %s\n", path);
      uint8_t zip_cd = 0;
      mz_zip_reader_get_zip_cd(zip_reader, &zip_cd);
      printf("Central directory %s zipped\n", (zip_cd) ? "is" : "is not");
      if (mz_zip_reader_goto_first_entry(zip_reader) == MZ_OK) {
        do {
          mz_zip_file *file_info = NULL;
          if (mz_zip_reader_entry_get_info(zip_reader, &file_info) != MZ_OK) {
            printf("Unable to get zip entry info\n");
            break;
          }
          printf("Zip entry %s\n", file_info->filename);
          if (mz_zip_reader_entry_open(zip_reader) == MZ_OK) {
            int32_t buf_size = (int32_t)mz_zip_reader_entry_save_buffer_length(zip_reader);
            char *buf = (char *)malloc(buf_size);
            int32_t err = mz_zip_reader_entry_save_buffer(zip_reader, buf, buf_size);
            if (err == MZ_OK) {
                printf("Bytes read from entry %d\n", buf_size);
            }else{
                printf("read from entry failed");
            }
            mz_zip_reader_entry_close(zip_reader);
          }
        } while (mz_zip_reader_goto_next_entry(zip_reader) == MZ_OK);
      }
      // 在zip文件中查找指定名称的文件
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
