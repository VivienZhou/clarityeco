#DjiNN#
[DjiNN](http://djinn.clarity-lab.org/djinn/) is a unified Deep Neural Network (DNN) web-service that allows DNN based applications to make requests to DjiNN.
Used by [Tonic Suite](http://djinn.clarity-lab.org/tonic-suite/)


###Basic Setup Before Startup###

Some preliminary steps are need before running djinn. They are listed on the [tonic-common](
https://github.com/Lilisys/clarityeco/tree/mergeTonic/tonic-common
) page.
**Make sure you follow the tonic-common installation instructions before attempting to run DjiNN**

Once you've completed the installation, you're ready to run DjiNN.

###Running Djinn###
In a separate terminal, run an instance of the command center: _This section is still under construction_

In this terminal, run these commands from the `djinn/` directory:
```
$ make
$ ./DjinnService
```

Now you can run services such as [Tonic Image Processing](https://github.com/Lilisys/clarityeco/tree/imgMerge/tonic-img) or [Tonic Natural Language Processing](https://github.com/Lilisys/clarityeco/tree/nlpMerge).
