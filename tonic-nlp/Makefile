COMMON=../tonic-common
NLP-COMMON=common-nlp
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

ifeq ($(CPU_ONLY), 1)
    CXX_FLAGS  += -DCPU_ONLY
endif

LINK_FLAGS = $(LIBRARY_DIRS) \
                         $(COMMON)/libtonic.a \
                         -lrt \
                         -lboost_program_options \
                         -lboost_filesystem \
                         -lboost_system \
                         -lthrift \
                         -lboost_thread \
						 -lglog \
                         -lpthread \
                         $(CAFFE)/lib/libcaffe.so

SRC=common-nlp/src
GEN=../gen-cpp
LIBS=-lthrift 


#File Names
NLP_SERVICE  	= NLPService		
NLP_CLIENT  	= NLPClient
SOURCES = $(wildcard $(SRC)/*.cpp) $(wildcard $(GEN)/*.cpp)
NLP_SERVICE_OBJ = $(SOURCES:.cpp=.o) NLPService.o		
NLP_CLIENT_OBJ 	= $(SOURCES:.cpp=.o)  NLPTestClient.o

# Main target
all: $(NLP_SERVICE) $(NLP_CLIENT)

$(NLP_SERVICE): $(NLP_SERVICE_OBJ) NLPService.cpp;		
	$(CXX) $(NLP_SERVICE_OBJ) -o $(NLP_SERVICE) $(CXX_FLAGS) $(LINK_FLAGS)		

$(NLP_CLIENT): $(NLP_CLIENT_OBJ) NLPTestClient.cpp;		
	$(CXX) $(NLP_CLIENT_OBJ) -o $(NLP_CLIENT) $(CXX_FLAGS) $(LINK_FLAGS)


# To obtain object files
%.o: %.cpp Makefile ;  $(CXX) -I$(INCLUDE_DIRS) -c $(CXX_FLAGS) $(EXTRA_FLAGS) $< -o $@

# To remove generated files
clean: ; rm -f $(NLP_SERVICE) $(NLP_CLIENT) $(GEN)/*.o $(COMMON)/src/*.o *.o
