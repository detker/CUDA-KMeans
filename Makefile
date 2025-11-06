NVCC        = nvcc
TARGET      = out
SRC         = kernel.cu
NVCCFLAGS   = -Xcompiler -fPIC -arch=sm_61
LDFLAGS     = 

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET) *.o

