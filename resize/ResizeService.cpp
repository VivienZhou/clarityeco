#include "opencv2/opencv.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/socket.h>
#include <string>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <map>
#include <glog/logging.h>

#include "boost/thread.hpp"
#include "boost/bind.hpp"
#include "boost/program_options.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string/replace.hpp"
#include "boost/lexical_cast.hpp"

#include "../gen-cpp/IPAService.h"
#include "../gen-cpp/SchedulerService.h"
#include "../gen-cpp/service_constants.h"
#include "../gen-cpp/service_types.h"
#include "../gen-cpp/types_constants.h"
#include "../gen-cpp/types_types.h"
#include "../gen-cpp/commons.h"
#include "../gen-cpp/ServiceTypes.h"
#include "../tonic-img/common-img/src/serialize.h"

#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TNonblockingServer.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TTransportUtils.h>
#include <thrift/TToString.h>


using namespace std;
using namespace cv;
using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

namespace fs = boost::filesystem;
namespace po = boost::program_options;

using boost::shared_ptr;

#define SERVICE_INPUT_TYPE "image"

ServiceTypes svt;

class ResizeServiceHandler : virtual public IPAServiceIf {
 public:
  ResizeServiceHandler() {
    this->SERVICE_NAME = svt.RESIZE_SERVICE;
    this->SCHEDULER_IP = "localhost";
    this->SCHEDULER_PORT = 8888;
    this->SERVICE_IP = "localhost";
    this->SERVICE_PORT = 6060;
  }

  //query.name == SERVICE_INPUT_TYPE
  void submitQuery(std::string& _return, const  ::QuerySpec& query) {
    time_t rawtime;
    time(&rawtime);
    LOG(INFO) << "receiving image data at " << ctime(&rawtime);

    int64_t start_time, end_time;
    
    for (int i = 0; i < query.content.size(); i++){
      struct timeval now;
      gettimeofday(&now, NULL);
      start_time = (now.tv_sec*1E6+now.tv_usec) / 1000;
      string str = img_resize(query.content[i]);
      _return += str;
      gettimeofday(&now, 0);
      end_time = (now.tv_sec*1E6+now.tv_usec) / 1000;
    }

    LatencyStat latencyStat;
    THostPort hostport;
    hostport.ip = this->SERVICE_IP;
    hostport.port = this->SERVICE_PORT;
    latencyStat.hostport = hostport;
    latencyStat.latency = end_time - start_time;
    this->scheduler_client->updateLatencyStat(this->SERVICE_NAME, latencyStat);
    LOG(INFO) << "update the command center latency statistics (" << latencyStat.latency << "ms)" << endl;
  }



void initialize(po::variables_map vm) {
    this->SERVICE_PORT      = vm["port"].as<int>();
    this->SERVICE_IP        = vm["svip"].as<string>();
    this->SCHEDULER_PORT    = vm["ccport"].as<int>();
    this->SCHEDULER_IP      = vm["ccip"].as<string>();

    TClient tClient;
    this->scheduler_client = tClient.creatSchedulerClient(this->SCHEDULER_IP, this->SCHEDULER_PORT);
    THostPort hostPort;
    hostPort.ip = this->SERVICE_IP;
    hostPort.port = this->SERVICE_PORT;
    RegMessage regMessage;
    regMessage.app_name = this->SERVICE_NAME;
    regMessage.endpoint = hostPort;
    LOG(INFO) << "registering to command center running at " << this->SCHEDULER_IP << ":" << this->SCHEDULER_PORT << endl;  
    this->scheduler_client->registerBackend(regMessage);
    LOG(INFO) << "service " << this->SERVICE_NAME << " successfully registered" << endl;
  }

string img_resize(const QueryInput &q_in){
  for(int i = 0; i < q_in.data.size(); ++i){
    int width = boost::lexical_cast<int>(q_in.tags[0]); //x size of img array
    int height = boost::lexical_cast<int>(q_in.tags[1]); //y size of img array
    string filename;
    
    LOG(INFO) << "Reading images...";
    stringstream ss;
    ss << q_in.data[i];
    Mat src = deserialize(ss, filename);

    //set destination size
    Mat dest(width, height, src.type());
    resize(src, dest, dest.size(), 0, 0);         
    LOG(INFO) << "new size: " << dest.rows << "x" << dest.cols << endl;
    return serialize(dest, filename);
  }
}

private:
    string SERVICE_NAME;
    string SERVICE_IP;
    int SERVICE_PORT;    
    string SCHEDULER_IP;
    int SCHEDULER_PORT;
    SchedulerServiceClient *scheduler_client;
     
};


po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
      ("port,p", po::value<int>()->default_value(6060),
          "Service port number")
      ("ccport,c", po::value<int>()->default_value(8888),
          "Command Center port number")
      ("ccip,i", po::value<string>()->default_value("localhost"),
          "Command Center IP address")
      ("svip,s", po::value<string>()->default_value("localhost"),
          "Service IP address");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    cout << desc << "\n";
    exit(1);
  }
  return vm;
}

int main(int argc, char **argv){
  po::variables_map vm = parse_opts(argc, argv);

  ResizeServiceHandler *ResizeService = new ResizeServiceHandler();
  boost::shared_ptr<ResizeServiceHandler> handler(ResizeService);
  boost::shared_ptr<TProcessor> processor(new IPAServiceProcessor(handler));
  
  TServers tServer;
  thread thrift_server;
  LOG(INFO) << "Starting the Resize service..." << endl;
  
  tServer.launchSingleThreadThriftServer(vm["port"].as<int>(), processor, thrift_server);
  ResizeService->initialize(vm);
  LOG(INFO) << "service Resize is ready..." << endl;
  thrift_server.join();

  return 0;
}


