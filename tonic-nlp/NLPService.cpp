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
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <string.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <glog/logging.h>
#include <stdint.h>
#include <ctime>
#include <cmath>
#include <boost/chrono/thread_clock.hpp>
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

#include <vector>
#include <map>
#include "caffe/caffe.hpp"

#include "common-nlp/src/SENNA_utils.h"
#include "common-nlp/src/SENNA_Hash.h"
#include "common-nlp/src/SENNA_Tokenizer.h"
#include "common-nlp/src/SENNA_POS.h"
#include "common-nlp/src/SENNA_CHK.h"
#include "common-nlp/src/SENNA_nn.h"
#include "common-nlp/src/SENNA_NER.h"

using caffe::Blob;
using caffe::Caffe;
using caffe::Net;
using caffe::Layer;
using caffe::shared_ptr;
using caffe::Timer;
using caffe::vector;

// import the Thread Safe Priority Queue
//#include "ThreadSafePriorityQueue.hpp"

using namespace std;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

using namespace apache::thrift;
using namespace apache::thrift::concurrency; 
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;
using namespace apache::thrift::server;

// define the constant
#define SERVICE_TYPE "nlp"
#define SERVICE_INPUT_TYPE "text"

bool debug;

class SennaServiceHandler : public IPAServiceIf {
	public:
		// put the model training here so that it only needs to
		// be trained once
		SennaServiceHandler() {
			this->SERVICE_NAME = SERVICE_TYPE;
			this->SCHEDULER_IP = "141.212.106.68";
			this->SCHEDULER_PORT = 8888;
			this->SERVICE_IP = "clarity07.eecs.umich.edu";
			this->SERVICE_PORT = 7070;
    	}

		~SennaServiceHandler() {
		}
		
		void submitQuery(std::string& _return, const ::QuerySpec& query) {
        senna_initialization();

        time_t rawtime;
        time(&rawtime);
        LOG(INFO) << "receiving nlp query at " << ctime(&rawtime);

        struct timeval now;
        gettimeofday(&now, NULL);
        int64_t start_time = (now.tv_sec*1E6+now.tv_usec) / 1000;
        _return = execute_senna(query.content[0].tags, query.content[0].data[0]);

        gettimeofday(&now, 0);
        int64_t end_time = (now.tv_sec*1E6+now.tv_usec) / 1000;
        LOG(INFO) << "the nlp is " << _return << endl;
    		
      	LatencyStat latencyStat;
     	  THostPort hostport;
      	hostport.ip = this->SERVICE_IP;
      	hostport.port = this->SERVICE_PORT;
      	latencyStat.hostport = hostport;
      	latencyStat.latency = end_time - start_time;
      	this->scheduler_client->updateLatencyStat(SERVICE_TYPE, latencyStat);
      	LOG(INFO) << "update the command center latency statistics (" << latencyStat.latency << "ms)" << endl;
    }
		

		void initialize(po::variables_map vm) {
			// 1. register to the command center
			this->SERVICE_PORT = vm["port"].as<int>();
			this->SERVICE_IP = vm["svip"].as<string>();
   		this->SCHEDULER_PORT = vm["ccport"].as<int>();
    	this->SCHEDULER_IP = vm["ccip"].as<string>();
    		
			TClient tClient;
			this->scheduler_client = tClient.creatSchedulerClient(this->SCHEDULER_IP, this->SCHEDULER_PORT);
			THostPort hostPort;
			hostPort.ip = this->SERVICE_IP;
			hostPort.port = this->SERVICE_PORT;
			RegMessage regMessage;
			regMessage.app_name = this->SERVICE_NAME;
			regMessage.endpoint = hostPort;
			LOG(INFO) << "registering to command center runnig at " << this->SCHEDULER_IP << ":" << this->SCHEDULER_PORT;	
			this->scheduler_client->registerBackend(regMessage);
      LOG(INFO) << "service " << this->SERVICE_NAME << " successfully registered";
		}

	private:
		SchedulerServiceClient *scheduler_client;

		/* SENNA Inits */
  	  char *opt_path;
  		int opt_usrtokens;
  		int *chk_labels;
  		int *pt0_labels;
  		int *pos_labels;
  		int *ner_labels;

  		/* inputs */
  		SENNA_Hash *word_hash;
  		SENNA_Hash *caps_hash;
 		  SENNA_Hash *suff_hash;
  		SENNA_Hash *gazt_hash;

  		SENNA_Hash *gazl_hash;
  		SENNA_Hash *gazm_hash;
  		SENNA_Hash *gazo_hash;
  		SENNA_Hash *gazp_hash;

  		/* labels */
  		SENNA_Hash *pos_hash;
  		SENNA_Hash *chk_hash;
  		SENNA_Hash *ner_hash;

  		// weights not used
 		  SENNA_POS *pos;
  		SENNA_CHK *chk;
  		SENNA_NER *ner;

  	  	/* tokenizer */
 		    SENNA_Tokenizer *tokenizer;
      	SENNA_Tokens *tokens;

      	TonicSuiteApp app;

	 	  QuerySpec newSpec;
		  string SERVICE_NAME;
		  string SCHEDULER_IP;
		  int SCHEDULER_PORT;
		  string SERVICE_IP;
		  int SERVICE_PORT;
		  string common;

    	void senna_initialization(){
      		opt_path = NULL;
        	opt_usrtokens = 0;

       		/* the real thing */
        	chk_labels = NULL;
        	pt0_labels = NULL;
        	pos_labels = NULL;
        	ner_labels = NULL;

        	/* inputs */
        	word_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/words.lst");
        	caps_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/caps.lst");
        	suff_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/suffix.lst");
        	gazt_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/gazetteer.lst");

       		gazl_hash = SENNA_Hash_new_with_admissible_keys(
            	opt_path, "common-nlp/hash/ner.loc.lst", "common-nlp/data/ner.loc.dat");
        	gazm_hash = SENNA_Hash_new_with_admissible_keys(
            	opt_path, "common-nlp/hash/ner.msc.lst", "common-nlp/data/ner.msc.dat");
        	gazo_hash = SENNA_Hash_new_with_admissible_keys(
            	opt_path, "common-nlp/hash/ner.org.lst", "common-nlp/data/ner.org.dat");
        	gazp_hash = SENNA_Hash_new_with_admissible_keys(
            	opt_path, "common-nlp/hash/ner.per.lst", "common-nlp/data/ner.per.dat");

        	/* labels */
        	pos_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/pos.lst");
        	chk_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/chk.lst");
        	ner_hash = SENNA_Hash_new(opt_path, "common-nlp/hash/ner.lst");

        	// weights not used
       		pos = SENNA_POS_new(opt_path, "common-nlp/data/pos.dat");
        	chk = SENNA_CHK_new(opt_path, "common-nlp/data/chk.dat");
        	ner = SENNA_NER_new(opt_path, "common-nlp/data/ner.dat");

        	/* tokenizer */
        	tokenizer =
         		SENNA_Tokenizer_new(word_hash, caps_hash, suff_hash, gazt_hash, gazl_hash,
                          gazm_hash, gazo_hash, gazp_hash, opt_usrtokens);  
    	}

    /*
     * execute_senna handles the request
     * vector<string> tags covers the comand line information parsed from the client side
     */
		string execute_senna(vector<string> tags, string input) {
      		app.task = "pos";
      		common = "../tonic-common/";
     	 	  string network = "pos.prototxt";
     	  	string weights = "pos.caffemodel";
      		app.djinn = false;
     	  	app.gpu = false;

      	for (int i = 0; i < tags.size(); i++){
       		if ((tags[i] == "--task")||(tags[i] == "-t")){
          		app.task = tags[++i];
        	}else if ((tags[i] == "--common")||(tags[i] == "-c"))
          		common = tags[++i];
       		else if ((tags[i] == "--network")||(tags[i] == "-n"))
          		network = tags[++i];
        	else if ((tags[i] == "--weight")||(tags[i] == "-w"))
          		weights = tags[++i];
        	else if ((tags[i] == "--djinn")||(tags[i] == "-d"))
          		app.djinn = tags[++i] == "1" ? true : false;
        	else if ((tags[i] == "--gpu")||(tags[i] == "-g"))
          		app.gpu = tags[++i] == "1" ? true : false;
        	else if ((tags[i] == "--debug") || (tags[i] == "-v"))
          		debug = tags[++i] == "1" ? true : false;
      	}

     	  app.weights = common + "weights/" + weights;
      	app.network = common + "configs/" + network;

     	if (app.djinn) {
        	app.hostname = SERVICE_IP;
        	app.portno = SERVICE_PORT;
      	} else {
        	app.net = new Net<float>(app.network);
        	app.net->CopyTrainedLayersFrom(app.weights);
        	if (app.gpu)
         	 	Caffe::set_mode(Caffe::GPU);
        	else
          	Caffe::set_mode(Caffe::CPU);
     	}

      	strcpy(app.pl.req_name, app.task.c_str());

      	// tokenize, read the file in
      	tokens = SENNA_Tokenizer_tokenize(tokenizer, input.c_str());
      	app.pl.num = tokens->n;
      	if (app.pl.num == 0) LOG(FATAL) << app.input << " empty or no tokens found.";

      	if (app.task == "pos")
       		POS_handler();
      	else if (app.task == "chk") {
        	CHK_handler();
      	} else if (app.task == "ner") {
        	NER_handler();
      	}

      	for (int i = 0; i < tokens->n; i++) {
        	printf("%15s", tokens->words[i]);
        	if (app.task == "pos"){
          		printf("\t%10s", SENNA_Hash_key(pos_hash, pos_labels[i]));
       		}else if (app.task == "chk")
          		printf("\t%10s", SENNA_Hash_key(chk_hash, chk_labels[i]));
        	else if (app.task == "ner")
          		printf("\t%10s", SENNA_Hash_key(ner_hash, ner_labels[i]));
        	printf("\n");
     	}

      	// clean up
      	SENNA_Tokenizer_free(tokenizer);
      	tokenizer = NULL;
        SENNA_POS_free(pos);
      	SENNA_CHK_free(chk);
      	SENNA_NER_free(ner);

      	SENNA_Hash_free(word_hash);
      	SENNA_Hash_free(caps_hash);
      	SENNA_Hash_free(suff_hash);
      	SENNA_Hash_free(gazt_hash);

      	SENNA_Hash_free(gazl_hash);
      	SENNA_Hash_free(gazm_hash);
      	SENNA_Hash_free(gazo_hash);
      	SENNA_Hash_free(gazp_hash);

      	SENNA_Hash_free(pos_hash);
      	SENNA_Hash_free(chk_hash);
     	  SENNA_Hash_free(ner_hash);

       	if (!app.djinn){
        	free(app.net);
       	}
       	printf("senna_all complete\n");
       	string str = "finish SENNA";
       	return str;
	}


    /*
     * Handle POS task
     */
    void POS_handler(){
      app.pl.size = pos->window_size *
        (pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size);

      if (app.djinn){
        SENNA_POS_forward_basic(pos, tokens->word_idx, tokens->caps_idx,
                                   tokens->suff_idx, app);

        //Using DjiNN service to process the data
      	string out_string = call_djinn(app.pl.req_name, app.pl.num * app.pl.size);
      	 
        //DjiNN service generates a series of floats, then converts them into a string
        //In POS_handler, it converts the string back into a series of floats.
        istringstream cin(out_string);
      	string buff;
       	for (int i = 0; i < app.pl.num * pos->output_state_size; i++){
        	cin>>buff;
        	float temp = boost::lexical_cast<float>(buff);
        	*(pos->output_state + i) = temp;
      	} 

        pos->labels = SENNA_realloc((void*)pos->labels, sizeof(int), app.pl.num);
        SENNA_nn_viterbi(pos->labels, pos->viterbi_score_init,
                   pos->viterbi_score_trans, pos->output_state,
                   pos->output_state_size, app.pl.num); 
        pos_labels = pos->labels;
      }else{
        reshape(app.net, app.pl.num * app.pl.size);
        SENNA_POS_forward_basic(pos, tokens->word_idx, tokens->caps_idx,
                                   tokens->suff_idx, app);
        pos_labels = SENNA_POS_forward_noDjiNN(pos, tokens->word_idx, tokens->caps_idx,
                                   tokens->suff_idx, app);
      }
      delete app.pl.data;
    }


    void CHK_handler(){
      /*Process the input text file using POS hanlder first
       *Then run CHK_handler using the result generated by POS handler
       *The algorithm is based on SENNA
      */

      app.pl.size = chk->window_size *
                  (chk->ll_word_size + chk->ll_caps_size + chk->ll_posl_size);

      TonicSuiteApp pos_app = app;
      pos_app.task = "pos";
      pos_app.network = common + "configs/" + "pos.prototxt";
      pos_app.weights = common + "weights/" + "pos.caffemodel";

      if (!pos_app.djinn) {
       pos_app.net = new Net<float>(pos_app.network);
       pos_app.net->CopyTrainedLayersFrom(pos_app.weights);
      }
      strcpy(pos_app.pl.req_name, pos_app.task.c_str());
      pos_app.pl.size = pos->window_size * (pos->ll_word_size + pos->ll_caps_size + pos->ll_suff_size);

     if (pos_app.djinn){
        SENNA_POS_forward_basic(pos, tokens->word_idx, tokens->caps_idx,tokens->suff_idx, pos_app);
        //Using DjiNN service to process the data
        string out_string = call_djinn(app.pl.req_name, app.pl.num * app.pl.size);
        //DjiNN service generates a series of floats, then converts them into a string
        //In POS_handler, it converts the string back into a series of floats.
        istringstream cin(out_string);
        string buff;
        for (int i = 0; i < pos_app.pl.num * pos->output_state_size; i++){
        	  cin>>buff;
          	float temp = boost::lexical_cast<float>(buff);
          	*(pos->output_state + i) = temp;
        } 

        pos->labels = SENNA_realloc(pos->labels, sizeof(int), pos_app.pl.num);
        SENNA_nn_viterbi(pos->labels, pos->viterbi_score_init,
                   pos->viterbi_score_trans, pos->output_state,
                   pos->output_state_size, pos_app.pl.num);
        pos_labels = pos->labels;

     }else{
        reshape(pos_app.net, pos_app.pl.num * pos_app.pl.size);
        SENNA_POS_forward_basic(pos, tokens->word_idx, tokens->caps_idx, tokens->suff_idx, pos_app);
        pos_labels = SENNA_POS_forward_noDjiNN(pos, tokens->word_idx, tokens->caps_idx,
                                   tokens->suff_idx, pos_app);
     }

     if (app.djinn){
        SENNA_CHK_forward_basic(chk, tokens->word_idx, tokens->caps_idx,
                                   pos_labels, app);
        //Using DjiNN service to process the datas
        string out_string = call_djinn(app.pl.req_name, app.pl.num * app.pl.size);
        //DjiNN service generates a series of floats, then converts them into a string
        //In POS_handler, it converts the string back into a series of floats.
        istringstream cin(out_string);
        string buff;
        for (int i = 0; i < app.pl.num * chk->output_state_size; i++){
           	cin>>buff;
         	float temp = boost::lexical_cast<float>(buff);
          	*(chk->output_state + i) = temp;
        } 

        chk->labels = SENNA_realloc(chk->labels, sizeof(int), app.pl.num);
        SENNA_nn_viterbi(chk->labels, chk->viterbi_score_init,
                   chk->viterbi_score_trans, chk->output_state,
                   chk->output_state_size, app.pl.num);
        chk_labels = chk->labels;
     }else{
      free(pos_app.net);
      reshape(app.net, app.pl.num * app.pl.size);
      SENNA_CHK_forward_basic(chk, tokens->word_idx, tokens->caps_idx,
                                   pos_labels, app);
      chk_labels = SENNA_CHK_forward_noDjiNN(chk, tokens->word_idx, tokens->caps_idx, pos_labels, app);
     }           
    }


    void NER_handler(){
      int input_size = ner->ll_word_size + ner->ll_caps_size + ner->ll_gazl_size +
                 ner->ll_gazm_size + ner->ll_gazo_size + ner->ll_gazp_size;
      app.pl.size = ner->window_size * input_size;

      if (app.djinn){
        SENNA_NER_forward_basic(ner, tokens->word_idx, tokens->caps_idx,
                               tokens->gazl_idx, tokens->gazm_idx,
                               tokens->gazo_idx, tokens->gazp_idx, app);
        //Using DjiNN service to process the data
        string out_string = call_djinn(app.pl.req_name, app.pl.num * app.pl.size);
        //DjiNN service generates a series of floats, then converts them into a string
        //In POS_handler, it converts the string back into a series of floats.
        istringstream cin(out_string);
        string buff;
        for (int i = 0; i < app.pl.num * ner->output_state_size; i++){
           	cin>>buff;
          	float temp = boost::lexical_cast<float>(buff);
          	*(ner->output_state + i) = temp;
        } 

        ner->labels = SENNA_realloc(ner->labels, sizeof(int), app.pl.num);
        SENNA_nn_viterbi(ner->labels, ner->viterbi_score_init,
                 ner->viterbi_score_trans, ner->output_state,
                 ner->output_state_size, app.pl.num);
        ner_labels = ner->labels;
       }else{
        reshape(app.net, app.pl.num * app.pl.size);
        SENNA_NER_forward_basic(ner, tokens->word_idx, tokens->caps_idx,
                                   tokens->gazl_idx, tokens->gazm_idx,
                                   tokens->gazo_idx, tokens->gazp_idx, app);
        ner_labels = SENNA_NER_forward_noDjiNN(ner, tokens->word_idx, tokens->caps_idx,
                                   tokens->gazl_idx, tokens->gazm_idx,
                                   tokens->gazo_idx, tokens->gazp_idx, app);  
      }
    }


    //call DjiNN service
    string call_djinn(string req_name, int in_len){
      //app.pl.data is the input data, which is a series of floats
      //convert the floats into a string
      //Then pass the string into DjiNN service as input
      string str;
      for (int i = 0; i < in_len; i++){
        float b = * ((float*)app.pl.data + i);
        string temp_str = boost::lexical_cast<std::string>(b);
        str = str + temp_str + " ";
      }

      string CC_SERVICE_IP = this->SCHEDULER_IP;
      int CC_SERVICE_PORT = this->SCHEDULER_PORT;
      ServiceTypes svt;
   
      LOG(INFO) << "Call Djinn Service";

      try{
        TClient tClient;
        SchedulerServiceClient * ccClient = NULL;
        IPAServiceClient * sennaClient = NULL;

        ccClient = tClient.creatSchedulerClient(CC_SERVICE_IP, CC_SERVICE_PORT);
        THostPort hostPort;
	      ccClient->consultAddress(hostPort, svt.DjiNN_SERVICE);
        LOG(INFO)<<"Command Center returns " << hostPort.ip << ":" << hostPort.port;
        sennaClient = tClient.creatIPAClient(hostPort.ip, hostPort.port);
      
        QuerySpec query;
        query.__set_name(SERVICE_INPUT_TYPE);
        QueryInput djinnQueryInput;
        vector<string> data;
        data.push_back(str);
        djinnQueryInput.__set_data(data);

        vector<string> tags;
        tags.push_back(req_name);
        std::ostringstream ss;
        ss<<in_len;
        tags.push_back(ss.str());
        djinnQueryInput.__set_tags(tags);

        vector<QueryInput> content;
        content.push_back(djinnQueryInput);
        query.__set_content(content);

        string out_string;
        sennaClient->submitQuery(out_string, query);
        return out_string;
      }catch (TException& tx) {
        LOG(FATAL)<<"invalid operation";    
      }
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
      ("ccip,i", po::value<string>()->default_value("141.212.106.68"),
          "Command Center IP address")
      ("svip,s", po::value<string>()->default_value("141.212.106.68"),
          "Service IP address");

  po::variables_map vm;
  po::store(po::parse_command_line(ac, av, desc), vm);
  po::notify(vm);

  if (vm.count("help")) {
    LOG(INFO) << desc << "\n";
    exit(1);
  }
  return vm;
}


int main(int argc, char **argv){
    po::variables_map vm = parse_opts(argc, argv);

    int port = vm["port"].as<int>();
    LOG(INFO) << "NLP Service using port:" <<port;

    SennaServiceHandler *SennaService = new SennaServiceHandler();
    boost::shared_ptr<SennaServiceHandler> handler(SennaService);
    boost::shared_ptr<TProcessor> processor(new IPAServiceProcessor(handler));

    TServers tServer;
    thread thrift_server;
    LOG(INFO) << "Starting the SENNA service...";

    tServer.launchSingleThreadThriftServer(port, processor, thrift_server);
    SennaService->initialize(vm);
    LOG(INFO) << "service SENNA is ready...";
    thrift_server.join();
	return 0;
}