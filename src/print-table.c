// src/print-table.c
//
// Generic boxed table shared by the fork's sizing tools (the model-info layer
// table, the kv-fit cost breakdown). One implementation, compiled into
// libllama; consumers call it through the extern "C" declaration in llama-ext.h.
//
// Rendering contract (byte-identical to the old model-info helpers):
//   - thin box-drawing characters only: ┌─┬─┐ ├─┼─┤ └─┴─┘ │
//   - key column (row keys) is right-aligned, like layer numbers / "TOTAL"
//   - data cells are right-aligned; header cells are centered (lean left on odd)
//   - one space of padding between text and the border each side
//
// Column widths: by default auto-measured per column from its header/row-key/
// value strings (max content length + 2). Passing cell_widths (one entry per
// column, key column first) overrides the measurement — used where a table
// layout is established and must not shift (the model-info layer table).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <llama.h>   // LLAMA_API export macro (public C header)

// Pure C, self-contained: the public declaration lives in llama-ext.h so C++
// consumers get it with the rest of the fork API. This file keeps its own
// prototype to stay independent of that C++-staged header — keep them in sync.
LLAMA_API void print_table(FILE * out,
        int n_row, int n_col,
        const char ** labels,
        const char ** row_labels,
        const char * key_header,
        const long ** data,
        const int * cell_widths,
        bool totals_separated);

static size_t pt_strlen(const char * s) { return s ? strlen(s) : 0; }

static void pt_spaces(char * buf, size_t * pos, size_t n) {
    while (n-- > 0) buf[(*pos)++] = ' ';
}

static void pt_append(char * buf, size_t * pos, const char * s, size_t len) {
    if (len > 0) { memcpy(buf + *pos, s, len); (*pos) += len; }
}

static void pt_border_row(FILE * out, char * line, int n_cols, const int * widths,
                          const char * tl, const char * mid, const char * tr) {
    size_t p = 0;
    pt_append(line, &p, tl, 3);
    for (int c = 0; c < n_cols; ++c) {
        if (c > 0) pt_append(line, &p, mid, 3);
        const char dash[3] = {'\xE2', '\x94', '\x80'};   // ─
        for (int j = 0; j < widths[c]; ++j) pt_append(line, &p, dash, 3);
    }
    pt_append(line, &p, tr, 3);
    line[p] = '\0';
    fprintf(out, "%s\n", line);
}

LLAMA_API void print_table(FILE * out,
                           int n_row, int n_col,
                           const char ** labels,      // n_col column headers (may be NULL)
                           const char ** row_labels,  // n_row row keys, right-aligned (may be NULL)
                           const char * key_header,   // header cell of the key column (NULL/"") = empty
                           const long ** data,        // n_row x n_col numbers, right-aligned (may be NULL)
                           const int * cell_widths,   // n_col+1 widths; NULL = auto-measure
                           bool totals_separated)    // middle border before the last row (TOTAL)
{
    if (!out || n_col <= 0 || n_row < 0) return;

    const int n_cols = n_col + 1;   // key column + data columns

    // ---- measure the maximum content length of every cell per column ---------
    size_t * max_len = (size_t *)calloc((size_t)n_cols, sizeof(size_t));
    if (!max_len) return;

    {
        const size_t l = pt_strlen(key_header);
        if (l > max_len[0]) max_len[0] = l;
        for (int r = 0; r < n_row; ++r) {
            const size_t rl = row_labels ? pt_strlen(row_labels[r]) : 0;
            if (rl > max_len[0]) max_len[0] = rl;
        }
    }
    for (int c = 0; c < n_col; ++c) {
        const size_t hl = labels ? pt_strlen(labels[c]) : 0;
        if (hl > max_len[c + 1]) max_len[c + 1] = hl;
        if (data) {
            for (int r = 0; r < n_row; ++r) {
                char tmp[32];
                snprintf(tmp, sizeof(tmp), "%ld", data[r][c]);
                const size_t l = strlen(tmp);
                if (l > max_len[c + 1]) max_len[c + 1] = l;
            }
        }
    }

    // ---- cell width: auto-measured or explicit --------------------------------
    int * widths = (int *)calloc((size_t)n_cols, sizeof(int));
    if (!widths) { free(max_len); return; }
    for (int c = 0; c < n_cols; ++c) {
        widths[c] = cell_widths ? cell_widths[c] : (int)max_len[c] + 2;
    }

    // ---- allocate one line buffer big enough for the widest row --------------
    // The border rows are the widest consumer: every cell character is a 3-byte
    // UTF-8 box character, so budget per logical width in BYTES. The last byte
    // is reserved for the NUL terminator that the fprintf("%s") relies on.
    size_t cap = 32;
    for (int c = 0; c < n_cols; ++c) cap += (size_t)(3 * widths[c] + 4);
    char * line = (char *)malloc(cap);
    if (!line) { free(max_len); free(widths); return; }
    line[cap - 1] = '\0';

    // ---- header row: key header + labels, centered ----------------------------
    pt_border_row(out, line, n_cols, widths, "┌", "┬", "┐");
    {
        size_t p = 0;
        const char pipe[3] = {'\xE2', '\x94', '\x82'};   // │
        pt_append(line, &p, pipe, 3);
        for (int c = 0; c < n_cols; ++c) {
            const char * txt = "";
            size_t l = 0;
            if (c == 0) {
                txt = key_header ? key_header : "";
                l = pt_strlen(txt);
            } else {
                txt = labels ? labels[c - 1] : "";
                l = pt_strlen(txt);
            }
            const int pad_l = (widths[c] - (int)l) / 2;
            const int pad_r = widths[c] - (int)l - pad_l - 1;
            pt_spaces(line, &p, (size_t)(pad_l > 0 ? pad_l : 0));
            pt_append(line, &p, txt, l);
            pt_spaces(line, &p, (size_t)(pad_r > 0 ? pad_r : 0));
            line[p++] = ' ';
            pt_append(line, &p, pipe, 3);
        }
        line[p] = '\0';
        fprintf(out, "%s\n", line);
    }
    pt_border_row(out, line, n_cols, widths, "├", "┼", "┤");

    // ---- data rows: row key right-aligned, numbers right-aligned --------------
    for (int r = 0; r < n_row; ++r) {
        size_t p = 0;
        const char pipe[3] = {'\xE2', '\x94', '\x82'};   // │
        pt_append(line, &p, pipe, 3);

        {
            const char * txt = row_labels ? (row_labels[r] ? row_labels[r] : "") : "";
            const size_t l = pt_strlen(txt);
            pt_spaces(line, &p, (size_t)widths[0] - 1 - l);
            pt_append(line, &p, txt, l);
            line[p++] = ' ';
            pt_append(line, &p, pipe, 3);
        }
        for (int c = 0; c < n_col; ++c) {
            char val[32];
            snprintf(val, sizeof(val), "%ld", data[r][c]);
            const size_t l = strlen(val);
            pt_spaces(line, &p, (size_t)widths[c + 1] - l - 1);
            pt_append(line, &p, val, l);
            line[p++] = ' ';
            pt_append(line, &p, pipe, 3);
        }
        line[p] = '\0';
        fprintf(out, "%s\n", line);

        if (totals_separated && r == n_row - 2) {   // separate the final TOTAL row
            pt_border_row(out, line, n_cols, widths, "├", "┼", "┤");
        }
    }

    pt_border_row(out, line, n_cols, widths, "└", "┴", "┘");

    free(line);
    free(widths);
    free(max_len);
}
