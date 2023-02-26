# General

- Add docs for all exported functions

## `io` contents

### `io._TiffImage`

- Enable fallthrough for bad tiles
- Use mmap and tile offsets from `libtiff` to decompose I/O from decoding to allow concurrent decoding.

### `ops.Mosaic`
