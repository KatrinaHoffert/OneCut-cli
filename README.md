## What is this?
It's a modification of the OneCut semi-automatic image segmentation program created by Lena Gorelick
(see README-original.txt). The original program was a GUI utility that was entirely standalone and
could save/load strokes to binary foreground and background images. This program made the following
changes to allow it to be used in an automated way in line with my research:

1. A greyscale strokes "label image" is loaded from the command line. This label image has one label
   (defaults to 1) for the foreground strokes and another label (defaults to 2) for the background
   strokes.
2. The labels are overrideable via command line options.
3. The GUI has been completely removed, making the program non-interactive.
4. The segmentation is run automatically, outputting to a user specified file (defaults to
   "<input file>_segmentated.png").
5. Now unnecesary code has been removed, files restructured.

## Download
If you're just interested in running the program without modifications, [prebuilt binaries are
available for 32 bit Windows](https://github.com/KatrinaHoffert/OneCut-cli/releases/download/1.0/OneCut-cli-1.0.zip).
The [Microsoft Visual Studio 2015 C++ redistributible](https://www.microsoft.com/en-ca/download/details.aspx?id=48145)
is required.

## Sample usage
This example uses the included sample files, "3063.jpg" and "3063-strokes.png":

    ./OneCut 3063.jpg 3063-strokes.png --fg-label 29 --bg-label 149 --output segmented.png

## Building
I've only ever tested this with Visual Studio 2015 (for which the project was update for). The provided
SLN should work fine for that. It requires the Windows 8.1 target SDK (but should retarget fine if
you have a different version). OpenCV 2 is required and is provided via Nuget. It probably will work
fine on non-Windows platforms and you'd just have to provide the OpenCV libraries and headers.

MSVS wants to copy all the OpenCV DLLs over, but the only needed ones are:

* `opencv_core310.dll`
* `opencv_imgcodecs310.dll`
* `opencv_imgproc310.dll`

Which makes up the release distribution of the program.

## Licensing
The original code and my changes uses the BSD 3 clause license (again, refer to README-original.txt).
The provided sample images are public domain.
