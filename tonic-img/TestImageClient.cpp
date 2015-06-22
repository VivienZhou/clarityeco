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
#include <glog/logging.h>

#include <stdlib.h>
#include <time.h>
#include <map>

// import opencv headers
#include "opencv2/core/core.hpp"
#include "opencv2/core/types_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/nonfree/gpu.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/gpu/gpu.hpp"
#include "boost/program_options.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string/replace.hpp"
#include "boost/lexical_cast.hpp"

// import the service headers
#include "../gen-cpp/IPAService.h"
#include "../gen-cpp/SchedulerService.h"
#include "../gen-cpp/service_constants.h"
#include "../gen-cpp/service_types.h"
#include "../gen-cpp/types_constants.h"
#include "../gen-cpp/types_types.h"
#include "../gen-cpp/commons.h"
#include "../gen-cpp/ServiceTypes.h"
#include "common-img/src/serialize.h"

// define the namespace
using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

using namespace apache::thrift;
using namespace apache::thrift::concurrency;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

//extracts cmd args for the client and pushes others to tags for service
string set_tags(int argc, char** argv, vector<string> &tags,
                string &ccip, int &ccport){
    string ret;
    for(int i = 1; i < argc; ++i){
        if(!strcmp(argv[i],"--file") || !strcmp(argv[i], "-f")){
            ret = argv[i + 1];
            ++i;
        }
        else if (!strcmp(argv[i],"--ccip") || !strcmp(argv[i], "-i")){
            ccip = argv[i + 1];
            ++i;
        }
        else if (!strcmp(argv[i],"--ccport") || !strcmp(argv[i], "-c")){
            ccport = atoi(argv[i + 1]);
            ++i;
        }
        else{
            tags.push_back(argv[i]);
        }
    }
    return ret;
}


int main(int argc, char** argv){
    String CC_SERVICE_IP = "localhost";
    int CC_SERVICE_PORT = 8888;
    ServiceTypes svt;
    vector<string> tags;

    string filename = set_tags(argc, argv, tags, CC_SERVICE_IP, CC_SERVICE_PORT);

    try{
        TClient tClient;
        SchedulerServiceClient * ccClient = NULL;
        IPAServiceClient * imgClient = NULL;
        //register with command center
        ccClient = tClient.creatSchedulerClient(CC_SERVICE_IP, CC_SERVICE_PORT);
        THostPort hostPort;
        ccClient->consultAddress(hostPort, svt.IMP_SERVICE);
        cout<<"Command Center returns: " << svt.IMP_SERVICE << " at " << 
                             hostPort.ip << ":" << hostPort.port<<endl;
        imgClient = tClient.creatIPAClient(hostPort.ip, hostPort.port);
        cout<<"hostPort.ip = " << hostPort.ip << "hostPort.port = " << hostPort.port<<endl;

        //set up the queryspec
        QuerySpec query;
        query.__set_name("test-img");

        string type = svt.SERVICE_INPUT_IMAGE;
        
        vector<string> data;

        QueryInput impQueryInput;
        
        impQueryInput.__set_type(type);
        impQueryInput.__set_tags(tags);
        LOG(INFO) << "Successfully parsed and set tags";

        //loop through all files listed and set data
        ifstream file(filename.c_str());
        string img_file;
        while (getline(file, img_file)) {
            LOG(INFO) << "filename: " << img_file;
            Mat img = imread(img_file);

            string instring = serialize(img, img_file);
        
            data.push_back(instring);
        }
        impQueryInput.__set_data(data);
        LOG(INFO) << "Successfully opened file and set input";

        query.content.push_back(impQueryInput);

        string outstring;
        LOG(INFO) << "Calling submit query... "<< endl;
     
        imgClient->submitQuery(outstring, query);
        cout<< outstring <<endl;
    }catch (TException& tx) {
        cout << "Could not talk to image" << endl;    
    }

    return 0;
}
