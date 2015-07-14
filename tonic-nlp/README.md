# ClarityEco/NLP Service

##Introduction
NLP Service is based on the language processing software SENNA. In order to improve the efficiency of this service, users are allowed to connnect the service with DjiNN, an open infrastructure for DNN. DjiNN is a service of the clarityeco system as well.<br>

##Components
* ```/tonic-nlp```
  * NLPService.cpp
  * NLPServiceClient.cpp
  * ```/common-nlp```
    * ```/src``` (source files for SENNA)
    * ```/data```(data for SENNA)
    * ```/hash```(data for SENNA)
    * ```/input```(input text files)

##Usage
### Setup Before Startup
#####Please make sure you follow the instructions on the [tonic-common page](https://github.com/Lilisys/clarityeco/tree/mergeTonic/tonic-common)<br>

## Communicate with Command Center
Plase make sure that you run an instance of command center in a separate terminal<br>
In ```/tonic-nlp```, <br>
```$ make```<br>
``` $ ./NLPService --port XXXX --svip XXXX --ccport XXXX --ccip XXXX```<br>
port --- Service port number<br>
svip --- Service IP address<br>
ccport --- Command Center port number<br>
ccip --- Command Center IP address<br>
IP addresses and port numbers depend on your servers and choices.<br>

##### NLP Service without DjiNN service
``` $ ./NLPClient --task pos --network pos.prototxt --weight pos.caffemodel --input common-nlp/input/small-input.txt --ccport XXXX --ccip XXXX```
##### NLP Service using DjiNN service
(Run DjiNN service first)<br>
``` $ ./NLPClient --task pos --network pos.prototxt --weight pos.caffemodel --input common-nlp/input/small-input.txt --djinn 1 --ccport XXXX --ccip XXXX```

