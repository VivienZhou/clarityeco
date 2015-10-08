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
#include "../tonic-img/common-img/src/serialize.h"

// define the namespace
using namespace std;
using namespace cv;

namespace fs = boost::filesystem;

using namespace apache::thrift;
using namespace apache::thrift::concurrency;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;


string get_input_file(int argc, char** argv){
    string ret;
    for(int i = 1; i < argc; ++i){
        if(!strcmp(argv[i],"--input") || !strcmp(argv[i], "-i")){
            ret = argv[i + 1];

        }
    }
    return ret;
}

int main(int argc, char** argv){
    String CC_SERVICE_IP = "141.212.106.68";
    int CC_SERVICE_PORT = 8888;
    ServiceTypes svt;

    try{
        TClient tClient;
        SchedulerServiceClient * ccClient = NULL;
        IPAServiceClient * resizeClient = NULL;
        //register with command center
        ccClient = tClient.creatSchedulerClient(CC_SERVICE_IP, CC_SERVICE_PORT);
        THostPort hostPort;
        ccClient->consultAddress(hostPort, svt.RESIZE_SERVICE);
        cout<<"Command Center returns: " << svt.RESIZE_SERVICE << " at " << 
                             hostPort.ip << ":" << hostPort.port<<endl;
        resizeClient = tClient.creatIPAClient(hostPort.ip, hostPort.port);
        cout<<"hostPort.ip = " << hostPort.ip << "hostPort.port = " << hostPort.port<<endl;

        //set up the queryspec
        QuerySpec query;
        query.__set_name("test-img");

        string type = svt.SERVICE_INPUT_IMAGE;
        vector<string> tags;
        vector<string> data;

        QueryInput impQueryInput;
        string filename = get_input_file(argc, argv);
        impQueryInput.__set_type(type);
        
        //testing resize for imc
        tags.push_back("277");
        tags.push_back("277");
        tags.push_back(filename);
        
        impQueryInput.__set_tags(tags);
        LOG(INFO) << "Successfully set tags";

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
     
        resizeClient->submitQuery(outstring, query);

        //Uncomment to save the image to disk for testing
        /*
        stringstream temp;
        temp << outstring;
        string temp1;

        Mat out = deserialize(temp, temp1);
        temp1 += "_new";
        try {
            imwrite(temp1, out);
        }
        catch (runtime_error& ex) {
            fprintf(stderr, "Exception converting image to PNG format: %s\n", ex.what());
            return 1;
        }*/

    }catch (TException& tx) {
        cout << "Could not talk to image" << endl;    
    }

    return 0;
}
