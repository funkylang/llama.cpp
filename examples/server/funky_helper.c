/*
  Copyright (C) 2024 by
  Dipl.-Ing. Michael Niederle

  This program is free software; you can redistribute it and/or modify
  it under the terms of the GNU Library General Public License, version 2, or
  (at your option) under the terms of the GNU Lesser General Public License,
  version 3.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU Lesser (Library) General Public License for more details.

  For details of the GNU General Public License see the accompanying
  files LGPLv2.txt and LGLPv3.txt or
  http://www.gnu.org/licenses/lgpl-2.0.html
  http://www.gnu.org/licenses/lgpl-3.0.html
  or write to the
  Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include <string.h>
#include "llama.h"

void longest_common_part(
  // original code by "rajsanghavi9"
  // (https://www.geeksforgeeks.org/longest-common-substring-dp-29/)
  llama_token *s, int n,
  llama_token *t, int m,
  int *len_p,
  int *i_p,
  int *j_p
) {
    #ifdef __STDC_NO_VLA__
      int dp[2][16385];
    #else
      int dp[2][m + 1];
    #endif
    int len = 0;
    int best_i = 0;
    int best_j = 0;

    memset(dp, 0, sizeof(dp)); // this is necessary!
    for (int i = 1; i <= n; ++i) {
	for (int j = 1; j <= m; ++j) {
	    if (s[i - 1] == t[j - 1]) {
		int new_len = dp[(i - 1) % 2][j - 1] + 1;
		dp[i % 2][j] = new_len;
		if (new_len > len) {
		    len = new_len;
		    best_i = i;
		    best_j = j;
		}
	    }
	    else
		dp[i % 2][j] = 0;
	}
    }
    *len_p = len;
    *i_p = best_i-len;
    *j_p = best_j-len;
}
