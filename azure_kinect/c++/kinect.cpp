#include "pch.h" // use stdafx.h in Visual Studio 2017 and earlier
#include "kinect.h"

#pragma comment(lib, "k4a.lib")
#include <k4a/k4a.h>

#include <stdio.h>
#include <stdlib.h>

int Kinect_viewer() {
	uint32_t count = k4a_device_get_installed_count();
	if (count == 0)
	{
		printf("No k4a devices attached!\n");
		exit(-1);
	}

	// Open the first plugged in Kinect device
	k4a_device_t device = NULL;
	if (K4A_FAILED(k4a_device_open(K4A_DEVICE_DEFAULT, &device)))
	{
		printf("Failed to open k4a device!\n");
		exit(-1);
	}

	// Get the size of the serial number
	size_t serial_size = 0;
	k4a_device_get_serialnum(device, NULL, &serial_size);

	// Allocate memory for the serial, then acquire it
	char* serial = (char*)(malloc(serial_size));
	k4a_device_get_serialnum(device, serial, &serial_size);
	printf("Opened device: %s\n", serial);
	free(serial);

	// Configure a stream of 4096x3072 BRGA color data at 15 frames per second
	k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
	config.camera_fps = K4A_FRAMES_PER_SECOND_15;
	config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
	config.color_resolution = K4A_COLOR_RESOLUTION_3072P;

	// Start the camera with the given configuration
	if (K4A_FAILED(k4a_device_start_cameras(device, &config)))
	{
		printf("Failed to start cameras!\n");
		k4a_device_close(device);
		exit(-1);
	}

	int x;

	// Camera capture and application specific code would go here
	// Access the depth16 image
	k4a_capture_t capture;
	k4a_wait_result_t get_capture_result = k4a_device_get_capture(device, &capture, K4A_WAIT_INFINITE);
	if (get_capture_result == K4A_WAIT_RESULT_SUCCEEDED)
	{
		//k4a_image_t image = k4a_capture_get_depth_image(capture);
		k4a_image_t image = k4a_capture_get_color_image(capture);
		if (image != NULL)
		{
			printf(" | Depth16 res:%4dx%4d\n",
				k4a_image_get_height_pixels(image),
				k4a_image_get_width_pixels(image));

			x = k4a_image_get_height_pixels(image);
				
			// Release the image
			k4a_image_release(image);
		}
		else {
			printf("NULL\n");
		}
	}
	else {
		printf("Not Capture\n");
	}

	// Release the capture
	k4a_capture_release(capture);

	// Shut down the camera when finished with application logic
	k4a_device_stop_cameras(device);
	k4a_device_close(device);

	return x;
}