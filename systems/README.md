# Subject Systems

In our experiments, we used measurements of subject systems from two different works. In the following, we explain the subject systems and measurements taken from [Kaltenecker et al.][kaltenecker],followed by the ones from [Werner][werner].


## Supplemental Material by [Kaltenecker et al.][kaltenecker]

The following systems were taken from the supplemental material provided in the paper [Distance-Based Sampling of Software Configuration Spaces by Kaltenecker et al.][kaltenecker]. We only provide a short description of the systems here, please refer to the paper for the original descriptions.
For each configuration, the authors conducted 5 measurements and added more until a [coefficient of variation][cv] of less than 10% was reached, reaching a maximum of 10 measurements per configuration.

[kaltenecker]: https://www.se.cs.uni-saarland.de/publications/docs/KaGrSi+19.pdf
[cv]: https://en.wikipedia.org/wiki/Coefficient_of_variation



#### [7-zip](https://www.7-zip.org/)

7-zip (7Z) is a file archiver written in C++. It offers configuration options to specify output sizes, chose compression methods and wether single- or multithreading should be used.
The authors measured compression time for 7Z (version 9.20) of the Canterbury corpus6 on an Intel Xeon E5-2690 with 64 GB RAM (Ubuntu 16.04).

#### [BERKELEYDB-C (BDB-C)](https://docs.oracle.com/cd/E17276_01/html/api_reference/C/frame_main.html)

BERKELEYDB-C (BDB-C) is an embedded database engine
written in C. The authors consider options regarding the use of encryption, and the page and cache size. The response time of BDD-C (version 4.4.20) was measured for different read and write queries on a machine with an Intel Core 2 Quad
CPU 2.66 GHz and 4 GB RAM.

#### [Dune MGS](https://dune-project.org/)

DUNE MGS (DUNE) is a geometric multigrid solver based on the DUNE framework. The authors consider options that chose different smoothing algorithms, as well as options specifying the number of pre-smoothing and post-smoothing steps. With DUNE (version 2.2), the time required to solve Poisson’s equation on a machine with an Intel i5-4570 and 32GB RAM was recorded.

#### [Hipacc](https://hipacc-lang.org/)
HIPAcc SOLVER (HIPAcc) is an image processing framework
written in C++. 
It offers options that specify types of memory to use and the number of pixels to be calculated per thread. The authors measured the time needed to solve partial differential equations on a machin with an nVidia
Tesla K20 with 5 GB RAM and 2496 cores.

#### [JavaGC](https://www.oracle.com/webfolder/technetwork/tutorials/obe/java/gc01/index.html)
JavaGC, the garbage collector of the Java VM, provides configuration options, which disable the explicit garbage collection call, modify the adaptive
garbage collection boundary, and modify the policy size, among others.
The garbage collection time of JavaGC (Java version 1.8) was measured during the execution of the DaCapo benchmark suite7 on machines with Intel Xeon
E5-2690 and 64 GB RAM.

#### [LLVM](https://llvm.org/)

LLVM is a compiler infrastructure written in C++.
Considered options include options to enable dead code elimination, jump threading and inlining. Using the Clang frontend, LLVM's (version 2.7) compile time on a machine with an AMD Athlon64 Dual Core and 2 GB RAM was measured for the opt-tool
benchmark.  

#### [lrzip](https://github.com/ckolivas/lrzip)

lrzip is a file compressor. 
Among others, it offers options specifying the use of encryption and the compression level. Compression time was measured for a 632MB-file with lrzip (version 0.600) on a machine with AMD Athlon64 Dual Core and 2 GB RAM.

#### [Polly](https://polly.llvm.org/)

Polly is a loop optimizer that builds on LLVM.
Considered options include, e.g., an option to chose the tile size and whether code should be parallelized. Using version 3.9, the runtime was measured of executing the
gemm program from polybench on a machine with an Intel Xeon E5-2690 and 64 GB RAM,

#### [VP9](https://www.webmproject.org/docs/encoder-parameters/)

VPXENC (VP9) is a video encoder that uses the VP9 video
coding format (see below for VP8). The authors considered options such as to define the encoding bitrate, the number of threads to use and for the quality of the encoded video.
The encoding time was measured for encoding 2 seconds from the Big Buck Bunny trailer on an Intel Xeon E5-2690 and 64 GB RAM,

#### [x264](https://www.videolan.org/developers/x264.html)

X264 is a video encoder for the H.264 compression format. It offers options specifying the use of the default entropy encoder, the number of reference frames, and the number of frames for lookahead and ratecontrol.
The authors measured x264's time to encode the [Sintel trailer][sintel] on a machine with Intel Core Q6600 and 4 GB RAM.


---

### Additional Measurements
Two additionaly subject systems were used, which are originally described [here][nw-thesis]. This work also offers measurements for *libvpx*, which was already used by [Kaltenecker et al.][kaltenecker]. However, here, VP8 was chosen as workload instead of VP9.
There are also feature models provided for the chosen subject systems in the [original work][werner]. 
Alle measurements were performed on workstations with Intel Core i5-4590, 16GB of memory and SSDs for storage.
Energy consumption was monitored at 1Hz with IPT iPower P1 power meters.

#### [HSQLDB](http://hsqldb.org/)

The HyperSQL DataBase (HSQLDB) is a relational database written in Java. It offers options for logging, encryption, backup and transaction control.
For HSQLDB the energy consumption was metered for the execution of the non-concurrent scenarios in [PolePosition](http://www.polepos.org/) that are compatible with the Java database interface JDBC using HSQLDB 2.4.2.

#### [PostgreSQL](https://www.postgresql.org/)

PostgreSQL (PSQL) is a relational database written in C. It exposes several options for buffers and memory consumption, as well for write operations.
The energy consumption was measured while executing the same benchmark as for HSQLDB (version 11.2) for a configuration space of 864 configurations.


#### [VP8](https://www.webmproject.org/docs/encoder-parameters/)
VP8 is an encoder for the WebM format written in C. It is the predecessor of VP9 and it is shipped in the same [libvpx package][libvpx]. VP8 offers a number options for video quality and the threads to be used.
For VP8 (version 1.8.0), the energy was metered to encode the ["Sintel" trailer][sintel] in YUV4MPEG2 (*.y4m file format) format with 480p
resolution with 2736 configurations.


[sintel]: https://media.xiph.org/
[nw-thesis]: <https://www.se.cs.uni-saarland.de/theses/NiklasWernerMA.pdf> "Niklas Werner's Bachelor thesis"
[werner]: <https://www.se.cs.uni-saarland.de/theses/NiklasWernerMA.pdf> "Niklas Werner's Bachelor thesis"

[libvpx]: https://www.webmproject.org/code/

[md-mape]: mape/README.md
[md-subject-systems]: ./systems/README.md
[md-calibration]: calibration/README.md
[md-main]: ./README.md