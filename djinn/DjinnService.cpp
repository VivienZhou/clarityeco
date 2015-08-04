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

#include "../gen-cpp/IPAService.h"
#include "../gen-cpp/SchedulerService.h"
#include "../gen-cpp/service_constants.h"
#include "../gen-cpp/service_types.h"
#include "../gen-cpp/types_constants.h"
#include "../gen-cpp/types_types.h"
#include "../gen-cpp/commons.h"
#include "caffe/caffe.hpp"
#include "../tonic-common/src/tonic.h"
#include "../tonic-common/src/timer.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;


#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TNonblockingServer.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>
#include <thrift/transport/TTransportUtils.h>
#include <thrift/TToString.h>


using namespace std;
namespace po = boost::program_options;
using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

namespace fs = boost::filesystem;

using boost::shared_ptr;


#define SERVICE_TYPE "djinn"

class DjinnServiceHandler : virtual public IPAServiceIf {
 public:
  DjinnServiceHandler() {
    // Your initialization goes here
    this->SERVICE_NAME = SERVICE_TYPE;
    this->SCHEDULER_IP = "localhost";
    this->SCHEDULER_PORT = 8888;
    this->SERVICE_IP = "localhost";
    this->SERVICE_PORT = 7071;
  }

  void submitQuery(std::string& _return, const  ::QuerySpec& query) {
    time_t rawtime;
    time(&rawtime);
    LOG(INFO) << "receiving djinn data at " << ctime(&rawtime);

    int64_t start_time, end_time;

    for (int i = 0; i < query.content.size(); i++){
      struct timeval now;
      gettimeofday(&now, NULL);
      start_time = (now.tv_sec*1E6+now.tv_usec) / 1000;
      string str = request_handler(query.content[i]);
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
    this->scheduler_client->updateLatencyStat(SERVICE_TYPE, latencyStat);
    LOG(INFO) << "update the command center latency statistics (" << latencyStat.latency << "ms)" << endl;
  }

  void train(po::variables_map vm){
    Caffe::set_phase(Caffe::TEST);
    if (vm["gpu"].as<bool>())
      Caffe::set_mode(Caffe::GPU);
    else
      Caffe::set_mode(Caffe::CPU);

    // load all models at init
    ifstream file(vm["nets"].as<string>().c_str());
    string net_name;
    while (file >> net_name) {
      string net = vm["common"].as<string>() + "configs/" + net_name;
      Net<float>* temp = new Net<float>(net);
      const std::string name = temp->name();
      nets[name] = temp;
      std::string weights = vm["common"].as<string>() +
                            vm["weights"].as<string>() + name + ".caffemodel";
      nets[name]->CopyTrainedLayersFrom(weights);
    }
  }

  void initialize(po::variables_map vm) {
    this->SERVICE_PORT      = vm["port"].as<int>();
    this->SERVICE_IP        = vm["svip"].as<string>();
    this->SCHEDULER_PORT    = vm["ccport"].as<int>();
    this->SCHEDULER_IP      = vm["ccip"].as<string>();

    this->vm = vm;

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


private:
    string SERVICE_NAME;
    string SERVICE_IP;
    int SERVICE_PORT;    
    string SCHEDULER_IP;
    int SCHEDULER_PORT;
    SchedulerServiceClient *scheduler_client;
    po::variables_map vm;
    map<string, Net<float>*> nets;
    
string request_handler(const QueryInput &q_in){
   
  // 1. Client sends the application type
  // 2. Client sends the size of incoming data
  // 3. Client sends data

  //q_in tags:
  // 0. request name 
  // 1. length of data
  string reqname = q_in.tags[0];
  LOG(INFO) << "The reqname is "  << reqname;
  int sock_elts = atoi(q_in.tags[1].c_str());
  LOG(INFO) << "The data size is " << sock_elts;

  string data = q_in.data[0];

  map<string, Net<float>*>::iterator it = nets.find(reqname);
  if (it == nets.end()) {
    LOG(ERROR) << "Task " << reqname << " not found.";
    return (void*)1;
  } else
    LOG(INFO) << "Task " << reqname << " forward pass.";

  // reshape input dims if incoming data != current net config
  LOG(INFO) << "Elements received on socket " << sock_elts << "\n";

  reshape(nets[reqname], sock_elts); 
  int in_elts = nets[reqname]->input_blobs()[0]->count();
  int out_elts = nets[reqname]->output_blobs()[0]->count();
  float* in = (float*)malloc(in_elts * sizeof(float));
  float* out = (float*)malloc(out_elts * sizeof(float));
 
  
  LOG(INFO) << "Starting lexical_cast...";
  istringstream ins(data);
  string buff;
  for (int i = 0; i < in_elts; i++){
    ins >> buff;
    float temp = boost::lexical_cast<float>(buff);
    * (in + i) = temp;
    
  }

  // Main loop of the thread, following this order
  // 1. Receive input feature (has to be in the size of sock_elts)
  // 2. Do forward pass
  // 3. Send back the result
  // 4. Repeat 1-3

  LOG(INFO) << "Executing forward pass.";

  SERVICE_fwd(in, in_elts, out, out_elts, nets[reqname]);

  LOG(INFO) << "Writing to return string.";
  string str;
  for (int i = 0; i < out_elts; i++){
    float b = * (out + i);
    string temp_str = boost::lexical_cast<string>(b);
    str = str + temp_str + " ";
  }

  // Exit the thread

  free(in);
  free(out);
  return str;
}

void SERVICE_fwd(float* in, int in_size, float* out, int out_size,
                 Net<float>* net) {
    string net_name = net->name();
    STATS_INIT("service", "DjiNN service inference");
    PRINT_STAT_STRING("network", net_name.c_str());

    if (Caffe::mode() == Caffe::CPU)
     PRINT_STAT_STRING("platform", "cpu");
    else
     PRINT_STAT_STRING("platform", "gpu");

    float loss;
    vector<Blob<float>*> in_blobs = net->input_blobs();

    tic();
    in_blobs[0]->set_cpu_data(in);
    vector<Blob<float>*> out_blobs = net->ForwardPrefilled(&loss);

    PRINT_STAT_DOUBLE("inference latency", toc());

    STATS_END();

    if (out_size != out_blobs[0]->count())
        LOG(FATAL) << "out_size =! out_blobs[0]->count())";
    else
        memcpy(out, out_blobs[0]->cpu_data(), out_size * sizeof(float));
 }  
 
};

po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
      ("port,p", po::value<int>()->default_value(7071),
          "Service port number")
      ("ccport,c", po::value<int>()->default_value(8888),
          "Command Center port number")
      ("ccip,i", po::value<string>()->default_value("localhost"),
          "Command Center IP address")
      ("svip,s", po::value<string>()->default_value("localhost"),
          "Service IP address")
      ("common,m", po::value<string>()->default_value("../tonic-common/"),
          "Directory with configs and weights")
      ("nets,n", po::value<string>()->default_value("nets.txt"),
          "File with list of network configs (.prototxt/line)")
      ("weights,w", po::value<string>()->default_value("weights/"),
          "Directory containing weights (in common)")
      ("gpu,g", po::value<bool>()->default_value(false), "Use GPU?")
      ("debug,v", po::value<bool>()->default_value(false),
          "Turn on all debug");

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

  DjinnServiceHandler *DjinnService = new DjinnServiceHandler();
  boost::shared_ptr<DjinnServiceHandler> handler(DjinnService);
  boost::shared_ptr<TProcessor> processor(new IPAServiceProcessor(handler));
  
  TServers tServer;
  thread thrift_server;
  LOG(INFO) << "Starting the Djinn service..." << endl;
  
  tServer.launchSingleThreadThriftServer(vm["port"].as<int>(), processor, thrift_server);
  DjinnService->train(vm);
  DjinnService->initialize(vm);
  LOG(INFO) << "service Djinn is ready..." << endl;
  thrift_server.join();

  return 0;
}
