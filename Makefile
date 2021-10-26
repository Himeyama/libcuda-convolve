NVCC = nvcc
 CXX = /usr/bin/g++
  CC = /usr/bin/gcc
 OPT = -ccbin $(CC) \
	-gencode arch=compute_52,code=sm_52 \
	-gencode arch=compute_60,code=sm_60 \
	-gencode arch=compute_61,code=sm_61 \
	-gencode arch=compute_70,code=sm_70 \
	-gencode arch=compute_75,code=sm_75

libcuda-convolve.so: cuda-convolve.cu
	$(NVCC) $(OPT) --shared -Xcompiler -fPIC $^ -o $@

test: test.cpp
	$(NVCC) $(OPT) $^ -lcuda-convolve -L. -I. -o $@

install: libcuda-convolve.so
	install -s $^ $(libdir)
	cp cuda-convolve.hpp $(incdir)
