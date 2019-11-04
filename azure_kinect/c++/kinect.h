#pragma once

#ifdef KINECTLIBRARY_EXPORTS
#define KINECTLIBRARY_API __declspec(dllexport)
#else
#define KINECTLIBRARY_API __declspec(dllimport)
#endif#pragma once

extern "C" KINECTLIBRARY_API int Kinect_viewer();