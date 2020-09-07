NVCC = "$(shell which nvcc)"
FILE := occupancy_api_timer
INC = -I$(CUDA_INC) -I$(MGPU_INC) -I$(CUB_INC) $(BOOST_DEPS) $(OMP_DEPS) $(METIS_DEPS) -I.. -I../..

all:
	$(NVCC) $(FILE).cu $(INC) --expt-extended-lambda -o $(FILE)

clean:
	rm -f $(FILE)
