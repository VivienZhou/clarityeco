#Tonic Image Processing#
Part of [Tonic Suite](http://djinn.clarity-lab.org/tonic-suite/)

###Basic Setup Before Startup###
Some preliminary steps are need before running tonic-img:

1. **Make sure you follow the installation instructions on the [tonic-common](
https://github.com/Lilisys/clarityeco/tree/mergeTonic/tonic-common
) page**

2. Run these commands
```
$ cd common-img/
$ sudo ./get-flandmark.sh
```
Now you're ready to run Tonic Image. 

###Running Tonic Image###
In a separate terminal, run an instance of command center: _This section is still under construction_

In another separate terminal, run an instance of [DjiNN](https://github.com/Lilisys/clarityeco/tree/mergeTonic/djinn)

If you plan on using sizes that do not fit the [Image Constraints](https://github.com/clarityinc/clarityeco/wiki/Tonic-Image-Processing#sample-client), then run an instance of [Resize](https://github.com/Lilisys/clarityeco/tree/mergeTonic/djinn) in another separate terminal

In this terminal, run these commands from the `tonic-img/` directory:
```
$ make
$ ./IMGServer
```

In a separate terminal, cd into this directory and run any of these client commands:

For IMC:
```
$ ./testClient --task imc --file common-img/imc-list.txt --ccip localhost --ccport 8888
```
For FACE:
```
$ ./testClient --task face --file common-img/face-list.txt --ccip localhost --ccport 8888
```
For DIG:
```
$ ./testClient --task dig  --file common-img/dig-list.txt --ccip localhost --ccport 8888
```
