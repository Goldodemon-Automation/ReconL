/* include/reconl_ui.h - the ReconL frontend's C ABI.
 *
 * This is the surface Java (Panama FFM), Kotlin, C, C++ and other Zig hosts
 * drive to build animated UIs that ReconL itself renders. It is deliberately
 * independent of reconl.h: a consumer needs exactly this file, and the
 * renderer stays an implementation detail behind `reconlUiRender`.
 *
 * Model: immediate mode. Per frame:
 *
 *   reconlUiBegin(ui, &input);          // feed pointer/keys once
 *   ...widgets...                       // each call draws and answers
 *   reconlUiEndFrame(ui);               // closes leaked panels, returns mesh?
 *   reconlUiRender(ui, pixels, size);   // ReconL renders it into RGBA8
 *
 * Identity is imgui-style: `id` scopes strings (use "Label##id" so two
 * identical labels keep separate hover/click state). Every animation advances
 * by `input.dt_ms` - never by a wall clock - so the same inputs twice produce
 * the same frames, which is what makes a golden a golden.
 *
 * Threading: one ReconLUi* belongs to one thread. The library keeps no
 * global mutable state beyond a creation-error slot read by
 * `reconlUiLastError(NULL)`.
 *
 * Strings are NUL-terminated UTF-8. Only Latin text is shaped (see
 * font.zig for the rationale).
 */
#ifndef RECONL_UI_H
#define RECONL_UI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#  ifdef RECONLUI_BUILD_SHARED
#    define RECONLUI_API __declspec(dllexport)
#  elif defined(RECONLUI_USE_SHARED)
#    define RECONLUI_API __declspec(dllimport)
#  else
#    define RECONLUI_API
#  endif
#else
#  define RECONLUI_API
#endif

/* Bump on any breaking change to a struct or signature below. */
#define RECONLUI_ABI_VERSION 1u

/* Surface dimensions a host asks for at creation; RGBA8, top-down. */
typedef struct ReconLUi ReconLUi; /* opaque */

/* One frame of input. `down` is the pointer state *now*; `pressed` /
 * `released` are the edges that happened during this frame. `wheel` counts
 * notches (+ = down). `dt_ms` drives every animation - 16.7 for 60 Hz. */
typedef struct ReconLUiInput {
    float    px;
    float    py;
    uint32_t down;
    uint32_t pressed;
    uint32_t released;
    float    wheel;
    float    dt_ms;
} ReconLUiInput;

/* Surface-pixel rectangle (y grows downward). */
typedef struct ReconLRect {
    float x;
    float y;
    float w;
    float h;
} ReconLRect;

/* Text roles - pairs of size + family from the theme's type scale. */
#define RECONLUI_ROLE_TITLE   0 /* Outfit, large   */
#define RECONLUI_ROLE_HEADING 1 /* Outfit, section */
#define RECONLUI_ROLE_BODY    2 /* Inter, default  */
#define RECONLUI_ROLE_LABEL   3 /* Inter, small    */
#define RECONLUI_ROLE_MONO    4 /* falls back to the UI face */

/* reconlUiButton flags. */
#define RECONLUI_BUTTON_PRIMARY 1u /* filled with the accent colour     */

/* reconlUiBeginPanel flags. */
#define RECONLUI_PANEL_SHADOW    1u /* stacked translucent drop shadow  */
#define RECONLUI_PANEL_NO_BORDER 2u /* suppress the default hairline     */

/* ---------------------------------------------------------------- lifecycle */

/* The ABI version this binary implements (see RECONLUI_ABI_VERSION). */
RECONLUI_API uint32_t reconlUiAbiVersion(void);

/* Creates a UI on a headless ReconL device, preferring the available hardware
 * backend and falling back to the reference backend, and loads the bundled
 * fonts. Returns NULL on failure; ask reconlUiLastError(NULL) why.
 * `width`/`height` are surface pixels (min 1). */
RECONLUI_API ReconLUi* reconlUiCreate(uint32_t width, uint32_t height);

/* Destroys everything the create call made. NULL is ignored. */
RECONLUI_API void reconlUiDestroy(ReconLUi* ui);

/* Recreates the swapchain for a new surface size and resizes layout.
 * Failing leaves the old size intact; returns 0 on success. A successful
 * resize discards the previous frame mesh; begin/build/endFrame again before
 * rendering to redraw at the new size. */
RECONLUI_API int32_t reconlUiResize(ReconLUi* ui, uint32_t width, uint32_t height);

/* Sets the render-pass clear colour (the theme background by default).
 * Finite channels clamp to 0..1; non-finite channels become 0. */
RECONLUI_API void reconlUiSetClear(ReconLUi* ui, float r, float g, float b, float a);

/* The last error message, NUL-terminated. `ui` may be NULL to read a failed
 * reconlUiCreate's reason. Valid until the next failing call on that UI. */
RECONLUI_API const char* reconlUiLastError(ReconLUi* ui);

/* ------------------------------------------------------------------- frame */

/* Starts a frame with this input and clears last frame's geometry. */
RECONLUI_API void reconlUiBegin(ReconLUi* ui, const ReconLUiInput* input);

/* Pops one container (panel/row/scroll). The root frame cannot be popped. */
RECONLUI_API void reconlUiEnd(ReconLUi* ui);

/* Closes anything left open and freezes the frame's mesh for reconlUiRender.
 * Returns 1 when the mesh has geometry, 0 when the UI drew nothing. */
RECONLUI_API uint32_t reconlUiEndFrame(ReconLUi* ui);

/* Renders the last endFrame mesh through ReconL into `pixels` (RGBA8,
 * top-down, at least width*height*4 bytes; `pixels_size` is that length).
 * Returns 0 on success, negative on failure (message via reconlUiLastError).
 * An empty mesh fills the clear colour without touching the renderer. */
RECONLUI_API int32_t reconlUiRender(ReconLUi* ui, void* pixels, uint64_t pixels_size);

/* Number of successful reconlUiRender calls (empty-frame clears included). */
RECONLUI_API uint32_t reconlUiFrameIndex(ReconLUi* ui);

/* ----------------------------------------------------------------- widgets */

/* A run of text. In a column it wraps at the content width; `role` is a
 * RECONLUI_ROLE_*, `color` may be NULL for the theme's text colour.
 * Returns the height consumed. */
RECONLUI_API float reconlUiLabel(ReconLUi* ui, const char* text, int32_t role,
                                 const float* color);

/* A button. Returns 1 only on release over it (a full click). `id` scopes
 * state; "Save##a" and "Save##b" are two buttons that both read Save. */
RECONLUI_API uint32_t reconlUiButton(ReconLUi* ui, const char* id, const char* text,
                                     uint32_t flags);

/* An animated pill switch. `value` is in/out (0/1); returns 1 when this
 * frame's click flipped it. The knob is a spring - it keeps settling after
 * the click lands. */
RECONLUI_API uint32_t reconlUiToggle(ReconLUi* ui, const char* id, const char* text,
                                     uint32_t* value);

/* A labelled track; dragging sets `value` (clamped to min..max).
 * Returns 1 only when the value actually changed this frame. */
RECONLUI_API uint32_t reconlUiSlider(ReconLUi* ui, const char* id, const char* text,
                                     float* value, float min, float max);

/* An animated progress bar for t in 0..1 (visually smoothed). */
RECONLUI_API void reconlUiProgress(ReconLUi* ui, const char* id, float t, float height);

/* A hairline rule / a vertical gap in the column. */
RECONLUI_API void reconlUiSeparator(ReconLUi* ui);
RECONLUI_API void reconlUiSpace(ReconLUi* ui, float amount);

/* Opens a padded, bordered column at `rect`. Pairs with reconlUiEnd (or
 * reconlUiEndFrame if the host forgets - it closes leaked panels). */
RECONLUI_API void reconlUiBeginPanel(ReconLUi* ui, const ReconLRect* rect,
                                     float pad, float gap, uint32_t flags);

/* A horizontal row of fixed height inside the current column. */
RECONLUI_API void reconlUiBeginRow(ReconLUi* ui, float height, float gap);

/* A wheel-scrollable viewport of `height` px over `content_height` px of
 * content. Returns the fixed viewport rect. Widgets drawn inside use the
 * spring-eased content offset automatically and are clipped to that rect. */
RECONLUI_API ReconLRect reconlUiBeginScroll(ReconLUi* ui, const char* id,
                                            float height, float content_height);

/* A sparkline of `count` samples (values may be NULL if count is 0). */
RECONLUI_API void reconlUiSparkline(ReconLUi* ui, const float* values, uint32_t count,
                                    float height);

/* The rect the most recent slot-based widget claimed (surface pixels) -
 * anchor tooltips, carets or a host-drawn overlay to it. Zeroed before the
 * first widget of a frame. */
RECONLUI_API ReconLRect reconlUiLastRect(ReconLUi* ui);

/* -------------------------------------------------------------------- mesh */

/* The frozen mesh from the last reconlUiEndFrame: counts, then pointers to
 * `ReconLVertex` (48 bytes: pos3, normal3, uv2, color4) and uint32
 * triangles. Buffers are valid until the next reconlUiBegin. */
RECONLUI_API uint32_t reconlUiVertexCount(ReconLUi* ui);
RECONLUI_API uint32_t reconlUiIndexCount(ReconLUi* ui);
RECONLUI_API const void* reconlUiVertices(ReconLUi* ui);
RECONLUI_API const void* reconlUiIndices(ReconLUi* ui);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RECONL_UI_H */
