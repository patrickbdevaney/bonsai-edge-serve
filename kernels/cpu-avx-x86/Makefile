# x86 AVX2 / AVX-VNNI GEMV bench for the shared Bonsai packed format.
#
#   make            AVX2 + AVX-VNNI, no BLAS control row
#   make blas       adds the fp32 cblas_sgemv control row; point BLAS_LIB
#                   and BLAS_SO at any CBLAS (defaults suit an apt
#                   libopenblas-dev install)
#   make blas BLAS_LIB=-L$$HOME/miniconda3/envs/tool-use/lib \
#             BLAS_SO=-l:libopenblas.so.0
#                   links the conda OpenBLAS, which ships no cblas.h --
#                   the bench declares the two symbols it needs itself
#
# On CPUs without AVX-VNNI, drop -mavxvnni; the bench compiles its
# VPMADDUBSW ladder only and says so at startup.

CC      ?= cc
# -ffp-contract=off: same rounding order as the scalar reference, so
# validation is bit-exact. Costs nothing at DRAM-streaming shapes.
CFLAGS  ?= -O3 -mavx2 -mavxvnni -mfma -ffp-contract=off -fopenmp -Wall -Wextra
BLAS_LIB ?=
BLAS_SO  ?= -lopenblas

gemv_avx_bench: gemv_avx_bench.c ../common/bonsai_gemv.h
	$(CC) $(CFLAGS) -o $@ gemv_avx_bench.c -lm

blas: gemv_avx_bench.c ../common/bonsai_gemv.h
	$(CC) $(CFLAGS) -DUSE_BLAS -o gemv_avx_bench gemv_avx_bench.c \
	    $(BLAS_LIB) $(BLAS_SO) -lm

test: test_shapes.c gemv_avx_bench.c ../common/bonsai_gemv.h
	$(CC) $(CFLAGS) -o test_shapes test_shapes.c -lm
	./test_shapes

clean:
	rm -f gemv_avx_bench test_shapes

.PHONY: blas test clean
