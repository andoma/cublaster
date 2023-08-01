cublaster: cublaster.cpp
	$(CXX) -Wall -Werror -O2 -o $@ $< -lcublas -lcudart

