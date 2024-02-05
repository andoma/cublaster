CXXFLAGS += $(shell pkg-config --cflags --libs cublas-12.2)

cublaster: cublaster.cpp
	$(CXX) -Wall -Werror -O2 -o $@ $< $(CXXFLAGS) -lcublas -lcudart
