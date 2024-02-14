# General

- Global tile cache

## libtiff

- Enable fallthrough for bad tiles
- Use mmap and tile offsets from `libtiff` to decompose I/O from decoding to allow concurrent decoding.

## `ops.Mosaic`

- Design & expose functional API
