COMMON=../tonic-common
export LIBRARY_PATH=/usr/local/lib
export LD_LIBRARY_PATH=/usr/local/lib
include $(COMMON)/Makefile.config

CXX = g++
CXX_FLAGS  = -O3 \
                         -w \
                         -fpermissive \
                         -I$(COMMON)/include \
                         -I$(CAFFE)/include \
                         -I$(CUDA)/include \
                         -Iz \
                         -I/usr/local/include\
                         -g

ifeq ($(CPU_ONLY), 1)
    CXX_FLAGS  += -DCPU_ONLY
endif

LINK_FLAGS = $(LIBRARY_DIRS) \
                         $(COMMON)/libtonic.a \
                         -lopencv_core \
                         -lopencv_highgui \
                         -lopencv_imgproc \
                         -lopencv_nonfree \
                         -lopencv_flann \
                         -lopencv_objdetect \
                         -lopencv_features2d \
                         -lopencv_gpu \
                         -lrt \
                         -lboost_program_options \
                         -lboost_filesystem \
                         -lboost_system \
                         -lthrift \
                         -lboost_thread \
						 -lglog \
                         -lpthread \
                         $(CAFFE)/lib/libcaffe.so

GEN=../gen-cpp
LIBS=-lthrift 

# File names
DJINN_SERVICE 	= DjinnService

SOURCES =  $(wildcard $(GEN)/*.cpp) 
DJINN_SERVICE_OBJ 	= $(SOURCES:.cpp=.o) DjinnService.o

# Main target
$(DJINN_SERVICE): $(DJINN_SERVICE_OBJ) DjinnService.cpp ;
	  $(CXX) $(DJINN_SERVICE_OBJ)  -o $(DJINN_SERVICE)  $(CXX_FLAGS) $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile ;  $(CXX) -I$(INCLUDE_DIRS) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(DJINN_SERVICE) $(GEN)/*.o $(COMMON)/src/*.o *.o
