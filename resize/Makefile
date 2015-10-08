COMMON=../tonic-common
IMG-COMMON=../tonic-img/common-img
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
                         $(IMG-COMMON)/libflandmark_static.a \
                         $(CAFFE)/lib/libcaffe.so

SRC=../tonic-img/common-img/src
GEN=../gen-cpp
LIBS=-lthrift 

# File names
RESIZE_SERVICE		= ResizeService
RESIZE_CLIENT		= testResizeClient
SOURCES = $(wildcard $(SRC)/*.cpp) $(wildcard $(GEN)/*.cpp)
RESIZE_SERVICE_OBJ 	= $(SOURCES:.cpp=.o) ResizeService.o
RESIZE_CLIENT_OBJ 	= $(SOURCES:.cpp=.o) testResizeClient.o

# Main target
all: $(RESIZE_SERVICE) $(RESIZE_CLIENT)

$(RESIZE_SERVICE): $(RESIZE_SERVICE_OBJ) ResizeService.cpp;
	  $(CXX) $(RESIZE_SERVICE_OBJ) -o $(RESIZE_SERVICE) $(CXX_FLAGS)  $(LINK_FLAGS)

$(RESIZE_CLIENT): $(RESIZE_CLIENT_OBJ) testResizeClient.cpp;
	  $(CXX) $(RESIZE_CLIENT_OBJ) -o $(RESIZE_CLIENT) $(CXX_FLAGS)  $(LINK_FLAGS)

# To obtain object files
%.o: %.cpp Makefile ;  $(CXX) -I$(INCLUDE_DIRS) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# To remove generated files
clean:
	rm -f $(IMG_SERVICE) $(IMG_CLIENT) $(GEN)/*.o $(COMMON)/src/*.o *.o
