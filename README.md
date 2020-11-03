# nn-perceptron
Java implementation of neuralnetwork perceptron using java/c++, opencl implementation

Inspired from [this video](https://www.youtube.com/watch?v=bVQUSndDllU), a java implementation of a [perceptron](https://en.wikipedia.org/wiki/Perceptron).
There's also a native (C++) implementation and an OpenCL implementation for performance comparison.

A GUI showing a field of red/green flowers, and showing the evolution of the network classification results.

### Build
Requires those 2 dependencies
- [OpenGL support](https://github.com/mflament/opengl-support)
- [OpenCL support](https://github.com/mflament/opencl-support)

From command line
`mvn -Prelease package` 

### Launch
`java -jar target/release.jar [matrix | mt | native | opencl]`

Where the parametrer is the implmentation to use:
 matrix: Java single thraded matrix
 mt: Java multi threaded matrix
 native: C++ implementation (windows only)
 opencl: OpenCL implementation (requires OpenCL drivers for your system)
