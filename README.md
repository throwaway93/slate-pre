If libraries are set, make should work. Set SLATE_DIR environment variable to this directory if you want to run benchmarks.
After starting slate daemon, benchmarks run as normal after built. Slate daemon in this repo is in CLI-mode only, so you 
may want to background it or have an extra shell.

Requires C++11, libcudart, libcuda, libnvrtc...
