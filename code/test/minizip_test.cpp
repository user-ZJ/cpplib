#include <iostream>
#include "mz_zip_rw.h"

void *file_stream = NULL;
const char *path = "/home/dmai/tmp/models.zip";

mz_zip_reader_create(&zip_reader);
mz_stream_os_create(&file_stream);

err = mz_stream_os_open(file_stream, path, MZ_OPEN_MODE_READ);
if (err == MZ_OK) {
    err = mz_zip_reader_open(zip_reader, file_stream);
    if (err == MZ_OK) {
        printf("Zip reader was opened %s\n", path);
        mz_zip_reader_close(zip_reader);
    }
}

mz_stream_os_delete(&file_stream);
mz_zip_reader_delete(&zip_reader);