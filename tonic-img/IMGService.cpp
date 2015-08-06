#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <map>
#include <glog/logging.h>

#include "opencv2/opencv.hpp"
#include "boost/program_options.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/algorithm/string/replace.hpp"
#include "boost/lexical_cast.hpp"

#include "caffe/caffe.hpp"
#include "../tonic-common/src/tonic.h"
#include "common-img/src/align.h"
#include "common-img/src/serialize.h"

#include "../gen-cpp/IPAService.h"
#include "../gen-cpp/SchedulerService.h"
#include "../gen-cpp/service_constants.h"
#include "../gen-cpp/service_types.h"
#include "../gen-cpp/types_constants.h"
#include "../gen-cpp/types_types.h"
#include "../gen-cpp/commons.h"
#include "../gen-cpp/ServiceTypes.h"
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using boost::shared_ptr;

using namespace std;
using namespace cv;

namespace po = boost::program_options;
struct timeval tv1, tv2;

ServiceTypes svt;

class IMGServiceHandler : virtual public IPAServiceIf {
 public:
  IMGServiceHandler() {
    this->SERVICE_NAME      = svt.IMP_SERVICE;
    this->SCHEDULER_IP      = "localhost";
    this->SCHEDULER_PORT    = 8888;
    this->SERVICE_IP        = "localhost";
    this->SERVICE_PORT      = 8080;
    app.task                = "imc";
    align                   = true;
    haar                    = "common-img/data/haar.xml";
    flandmark               = "common-img/data/flandmark.dat";

  }

  void submitQuery(std::string& _return, const  ::QuerySpec& query) {
    time_t rawtime;
    time(&rawtime);
    LOG(INFO) << "receiving image query at " << ctime(&rawtime);

    int64_t start_time, end_time;

    for (int i = 0; i < query.content.size(); i++){
      struct timeval now;
      gettimeofday(&now, NULL);
      start_time = (now.tv_sec*1E6+now.tv_usec) / 1000;
      string str = exec_img(query.content[i]);
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
    this->scheduler_client->updateLatencyStat(svt.IMP_SERVICE, latencyStat);
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

private:
  TonicSuiteApp app;
  string SERVICE_NAME;
  string SCHEDULER_IP;
  int SCHEDULER_PORT;
  string SERVICE_IP;
  int SERVICE_PORT;
  SchedulerServiceClient *scheduler_client;
  bool align;
  string flandmark;
  string haar;

  //parse command line arguments from the client side
  void parse_tags(const std::vector<string> &tags){
    for(int i = 0; i < tags.size(); ++i){
      if (tags[i] == "--task"           || tags[i] == "-t")
        app.task        = tags[i + 1];
      else if (tags[i] == "--align"     || tags[i] == "-l")
        align           = (tags[i + 1] == "1")? true : false;
      else if (tags[i] == "--haar"      || tags[i] == "-a")
        haar        = tags[i + 1];
      else if (tags[i] == "--flandmark" || tags[i] == "-f")
        flandmark   = tags[i + 1];
      else{}
    }
  }

  void call_resize(Mat &img, string filename, string imgData){
    //cause preprocess to be called after this function
    if(app.task == "face"){
       align = true;
    }
    //resize for IMC, DIG
    else{
      try{
        TClient tClient;
        SchedulerServiceClient * ccClient = NULL;
        IPAServiceClient * resizeClient = NULL;
        //register with command center
        ccClient = tClient.creatSchedulerClient(this->SCHEDULER_IP, this->SCHEDULER_PORT);
        THostPort hostPort;
        ccClient->consultAddress(hostPort, svt.RESIZE_SERVICE);
        LOG(INFO) <<"Command Center returns: " << svt.RESIZE_SERVICE << " at " << 
                             hostPort.ip << ":" << hostPort.port<<endl;
        resizeClient = tClient.creatIPAClient(hostPort.ip, hostPort.port);
        LOG(INFO) <<"hostPort.ip = " << hostPort.ip << "hostPort.port = " << hostPort.port<<endl;

        //set up the queryspec
        QuerySpec query;
        QueryInput impQueryInput;

        string type = svt.SERVICE_INPUT_IMAGE;
        impQueryInput.__set_type(type);

        vector<string> tags;
        vector<string> data;

        //specific sizes for each task
        string lw_size;
        if (app.task == "imc") lw_size = "227";
        else if (app.task == "face") lw_size = "152";
        else if (app.task == "dig") lw_size = "28";

        tags.push_back(lw_size);
        tags.push_back(lw_size);
        tags.push_back(filename);
        
        impQueryInput.__set_tags(tags);
        LOG(INFO) << "Successfully set tags";
        
        data.push_back(imgData);
        impQueryInput.__set_data(data);
        LOG(INFO) << "Successfully set input";

        query.content.push_back(impQueryInput);

        string outstring;
        LOG(INFO) << "Resizing... "<< endl;
     
        resizeClient->submitQuery(outstring, query);

        stringstream temp;
        temp << outstring;
        string temp1;
        img = deserialize(temp, temp1);
        
      } catch (TException& tx) {
        LOG(ERROR) << "Could not talk to resize" << endl;    
      }
    }
  }

  string call_djinn(stringstream &ret, string & s){
    string outstring;

    try {
      TClient tClient;
      SchedulerServiceClient * ccClient = NULL;
      IPAServiceClient * imgClient = NULL;

      //Register with Command Center
      ccClient = tClient.creatSchedulerClient(this->SCHEDULER_IP, this->SCHEDULER_PORT);
      THostPort hostPort;
      ccClient->consultAddress(hostPort, svt.DjiNN_SERVICE);
      imgClient = tClient.creatIPAClient(hostPort.ip, hostPort.port);
      LOG(INFO) <<"Command Center returns " << hostPort.ip << ":" << hostPort.port;

      //set up the data query
      QuerySpec query;
      query.__set_name(svt.SERVICE_INPUT_IMAGE);

      LOG(INFO) << "Setting tags";

      QueryInput djinnQueryInput;
      vector<string> tags;
      string req_name = app.task;
      int length = app.pl.num * app.pl.size;
      stringstream ss;
      ss << length;
      string str = ss.str();
      tags.push_back(req_name);
      tags.push_back(str);
      djinnQueryInput.__set_tags(tags);

      LOG(INFO) << "Request Name: " << req_name;
      LOG(INFO) << "Length: " << app.pl.num * app.pl.size;
      LOG(INFO) << "Setting input data...";

        
      vector<string> dataVec;
      dataVec.push_back(s);

      djinnQueryInput.__set_data(dataVec);

      query.content.push_back(djinnQueryInput);
      
      imgClient->submitQuery(outstring, query);
      return outstring;

    } catch (TException& tx) {
        LOG(ERROR) << "Could not talk to Djinn" << endl;    
    }
  }

  string exec_img(const QueryInput &q_in){
    parse_tags(q_in.tags);
    app.pl.size = 0;
    
    // hardcoded for AlexNet
    strcpy(app.pl.req_name, app.task.c_str());
    if (app.task == "imc") app.pl.size = 3 * 227 * 227;
    // hardcoded for DeepFace
    else if (app.task == "face")
      app.pl.size = 3 * 152 * 152;
    // hardcoded for Mnist
    else if (app.task == "dig")
      app.pl.size = 1 * 28 * 28;
    else
      LOG(FATAL) << "Unrecognized task.\n";

    // read in images
    // cmt: using map, cant use duplicate names for images
    // change to other structure (vector) if you want to send the same exact
    // filename multiple times

    vector<pair<string, Mat> > imgs;
    app.pl.num = 0;
    int j = 0;
    for (int i = 0; i < q_in.data.size(); ++i) {
      LOG(INFO) << "Reading images...";
      stringstream ss;
      ss << q_in.data[i];
     
      string filename;
      Mat img = deserialize(ss, filename);
      
      //Convert to grayscale for dig
      if (app.task == "dig"){
        Mat gs_bgr(img.size(), CV_8UC1);
        cvtColor(img, gs_bgr, CV_BGR2GRAY);
        img = gs_bgr;
      }
      
      //Resize if neccessary
      if (img.channels() * img.rows * img.cols != app.pl.size){
        LOG(ERROR) << "resizing " << filename << " to correct dimensions.\n";
        call_resize(img, filename, q_in.data[i]);
      } 
      //aligns and resizes image for FACE
      if(app.task == "face" && align){
        LOG(INFO) << "aligning " << filename;
        preprocess(img, flandmark, haar);
        // comment in to save + view aligned image
        // imwrite(filename+"_a", img);
      }

      imgs.push_back(make_pair(filename, img));
      ++app.pl.num;  
    }

    if (app.pl.num < 1) LOG(FATAL) << "No images read!";
    
    string s;
    // prepare data into array
    vector<pair<string, Mat> >::iterator it;
    app.pl.data = (float*)malloc(app.pl.num * app.pl.size * sizeof(float));
    float* preds = (float*)malloc(app.pl.num * sizeof(float));
    int img_count = 0;
    for (it = imgs.begin(); it != imgs.end(); ++it) {
      int pix_count = 0;
      for (int c = 0; c < it->second.channels(); ++c) {
        for (int i = 0; i < it->second.rows; ++i) {
          for (int j = 0; j < it->second.cols; ++j) {
            Vec3b pix = it->second.at<Vec3b>(i, j);
            float* p = (float*)(app.pl.data);
            p[img_count * app.pl.size + pix_count] = pix[c];
            //setting up a string for queryInput            
            s += std::to_string(pix[c]) + " ";
            ++pix_count;
          }
        }
      }
      ++img_count;
    }
    
    stringstream ret;
    string outstring = call_djinn(ret, s);

    istringstream iss(outstring);
    string predString;
     
    for (it = imgs.begin(); it != imgs.end(); it++) {
      iss >> predString;
      ret << "Image: " << it->first << " class: " << predString << "\n";
      LOG(INFO) << "Image: " << it->first
                << " class: " << predString << endl;
      ;
      }
   
     
    free(app.pl.data);
    free(preds);

    return ret.str();
  }
};

//parse command line arguments from server side
po::variables_map parse_opts(int ac, char** av) {
  // Declare the supported options.
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "Produce help message")
      ("port,p", po::value<int>()->default_value(8080),
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


int main(int argc, char **argv) {
  po::variables_map vm = parse_opts(argc, argv);

  IMGServiceHandler *IMGService = new IMGServiceHandler();

  boost::shared_ptr<IMGServiceHandler> handler(IMGService);
  boost::shared_ptr<TProcessor> processor(new IPAServiceProcessor(handler));
    
  TServers tServer;
  thread thrift_server;
  LOG(INFO) << "Starting the IMP service..." << endl;
    
  tServer.launchSingleThreadThriftServer(vm["port"].as<int>(), processor, thrift_server);
  IMGService->initialize(vm);
  LOG(INFO) << "service IMP is ready..." << endl;
  thrift_server.join();

  return 0;
}


