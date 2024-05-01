#include <stdio.h>
#include <stdbool.h>
#include <sys/sysinfo.h>

int main(int argc, char **argv) {
  __builtin_cpu_init();
  bool has_avx = __builtin_cpu_supports("avx");
  bool has_avx2 = __builtin_cpu_supports("avx2");
  bool has_avx512 = __builtin_cpu_supports("avx512f");
  printf(
  "avx:  %s\n"
  "avx2: %s\n"
  "avx512: %s\n",
  has_avx ? "yes" : "no",
  has_avx2 ? "yes" : "no",
  has_avx512 ? "yes" : "no"
  );
  struct sysinfo info;
  if (sysinfo(&info) == 0) {
    printf("ram: %lu\n", info.totalram);
  }
}
