__kernel void convolution2D(
    __global float * inputData, __global float * outputData, __constant float * maskData,
    int width, int height, int maskWidth,  int imageChannels){
    
    //@@ Insert code to implement matrix multiplication here

    int i = get_global_id(0);
    int j = get_global_id(1);
    int maskRadius = maskWidth/2;
    
    //Loop through for the dot-product of matrix multiplication
    for (int k=0; k < imageChannels; k++){
        float accum = 0;
        //Loop over y pixels
        for (int y = -maskRadius; y <= maskRadius; y++){
            //loop over x pixels
            for (int x = -maskRadius; x <= maskRadius; x++){ 
                int xOffset = j+x;
                int yOffset = i+y;
                
                //Before multiplying, make sure the pixels exist! can't convolve edge of matrix pixels since nothing exists outside of the image.
                if ( ((xOffset>=0) && (xOffset<width)) && ((yOffset>=0) && (yOffset<height)) ){

                    float imagePixel = inputData[(yOffset * width + xOffset) * imageChannels + k];
                    float maskValue = maskData[(y + maskRadius) * maskWidth + x + maskRadius];
                    accum += imagePixel * maskValue;
                }
            }
        }

        if (accum<0){
            accum = 0;
        }
        else if (accum>1){
            accum = 1;
        }
        outputData[(i * width + j) * imageChannels + k] = accum;
    }
    

}