// import the thrift headers
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TTransportUtils.h>
#include <thrift/TToString.h>

// import common utility headers
#include <iostream>
#include <stdexcept>
#include <sstream>
#include <string>
#include <fstream>
#include <sys/time.h>
#include <iomanip>
#include <boost/filesystem.hpp>

#include <stdlib.h>
#include <time.h>
#include <map>

#include "boost/program_options.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string/replace.hpp"

// import the service headers
#include "../gen-cpp/IPAService.h"
#include "../gen-cpp/SchedulerService.h"
#include "../gen-cpp/service_constants.h"
#include "../gen-cpp/service_types.h"
#include "../gen-cpp/ServiceTypes.h"
#include "../gen-cpp/types_constants.h"
#include "../gen-cpp/types_types.h"
#include "../gen-cpp/commons.h"
#include <glog/logging.h>
// define the namespace
using namespace std;
namespace fs = boost::filesystem;

using namespace apache::thrift;
using namespace apache::thrift::concurrency;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

/* fgets max sizes */
#define MAX_TARGET_VB_SIZE 256

bool debug;


int main(int argc, char ** argv){
    string CC_SERVICE_IP = "141.212.106.68";
    int CC_SERVICE_PORT = 8888;
    
    ServiceTypes svt;

    try{
		TClient tClient;
		SchedulerServiceClient * ccClient = NULL;
		IPAServiceClient * sennaClient = NULL;;

   		QuerySpec query;
    	query.name = "test-senna";
    	QueryInput sennaQueryInput;

   	 	vector<string> tags;
        string input = "common-nlp/input/small-input.txt";
        for (int i = 1; i < argc; i++){
        	string str = argv[i];
            if (str == "--input"){
                input = argv[++i];
                continue;
            }else if (str == "--ccip"){
                CC_SERVICE_IP = argv[++i];
                continue;
            }else if (str == "--ccport"){
                CC_SERVICE_PORT = std::atoi(argv[++i]);
                continue;
            }
            tags.push_back(argv[i]);
        }

        ccClient = tClient.creatSchedulerClient(CC_SERVICE_IP, CC_SERVICE_PORT);
        THostPort hostPort;
        ccClient->consultAddress(hostPort, svt.NLP_SERVICE);
        LOG(INFO)<<"Command Center returns " << hostPort.ip << ":" << hostPort.port;
        sennaClient = tClient.creatIPAClient(hostPort.ip, hostPort.port);

        // read input file
        ifstream file(input.c_str());
        string str;
        string text;
        while (getline(file, str)) text += str;
        vector<string> data;
        data.push_back(text);
        sennaQueryInput.__set_data(data);
        sennaQueryInput.__set_tags(tags);

        vector<QueryInput> content;
        content.push_back(sennaQueryInput);
        query.__set_content(content);

        string outstring;
        sennaClient->submitQuery(outstring, query);
        LOG(INFO)<<outstring<<endl;
	}catch (TException& tx) {
     	LOG(FATAL)<<"invalid operation"<<endl;    
  	}

  	return 0;
}
