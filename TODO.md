# General

## openslide

- Upgrade binaries from 3.4.1 to 4.0 (`libopenslide-0.dll` -> `libopenslide-1.dll`)
- Check if `libopenslide==4.0` present in Ubuntu repos
- Enable ICC conversion for tiles

## libtiff

- Enable ICC conversion for tiles
- Enable fallthrough for bad tiles
- Use mmap and tile offsets from `libtiff` to decompose I/O from decoding to allow concurrent decoding.

## `ops.Mosaic`

- Design & expose functional API
