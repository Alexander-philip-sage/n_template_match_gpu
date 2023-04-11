export PKG_CONFIG_PATH=~/opencv_second_env_cpu/opencv_4.7.0_build/install/lib64/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:~/opencv_second_env_cpu/opencv_4.7.0_build/install/lib64
CC test_template_match.cpp -o tm_test.exe `pkg-config --cflags --libs opencv4`
./tm_test.exe
