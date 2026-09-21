/* ReconL ABI version.
 *
 * RECONL_ABI_VERSION is bumped on any breaking change to a struct layout or a
 * function signature. reconlCreateDevice refuses a Descriptor whose
 * struct_size/type do not match, and reports RECONL_ERR_ABI_VERSION when the
 * linked library's ABI version differs from the header's.
 *
 * Growth policy (see the struct rules in reconl.h): new fields arrive through
 * the `next` chain or at the end of a struct with a bumped struct_size. Fields
 * are never reordered, retyped, or removed in a minor release.
 */
#ifndef RECONL_VERSION_H
#define RECONL_VERSION_H

#define RECONL_VERSION_MAJOR 0
#define RECONL_VERSION_MINOR 1
#define RECONL_VERSION_PATCH 0

/* Packed as MAJOR * 10000 + MINOR * 100 + PATCH. */
#define RECONL_ABI_VERSION 100

#endif /* RECONL_VERSION_H */
