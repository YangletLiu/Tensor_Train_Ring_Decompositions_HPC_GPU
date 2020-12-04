cc=/usr/local/cuda-10.1/bin/nvcc
des= test
#source = de_cp.cpp cp_als.cu kernel.cu  
#source = de_cp4.cpp cp4.cu kernel.cu  
#source = de_tucker4.cpp tucker4_hosvd.cu kernel.cu
source = test.cu
#source = de_tucker.cpp tucker_hooi.cu kernel.cu  
link = -lcublas -std=c++11

$(des):$(source)
	$(cc) -g $(source) $(link) -o $(des) 

clean:
	rm -rf $(des)
