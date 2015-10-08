#Resize#
Resize is a service written using the [OpenCV](http://docs.opencv.org/modules/imgproc/doc/geometric_transformations.html#resize) image processing library that rescales images to desired dimensions.

Resize has been wraped in RPC framework using [Apache Thrift](http://thrift.apache.org/). 

###Running Resize###
In a separate terminal, run an instance of command center: _This section is still under construction_

In this terminal, run these commands from the `resize/` directory:
```
$ make
$ ./ResizeService
```

In a separate terminal, cd into this directory and run 
```
$ ./testResizeClient resizeTest.txt
```

