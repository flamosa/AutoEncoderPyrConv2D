import cv2 as cv
import numpy as np
import os
import tensorflow as t
import matplotlib.pyplot as plt

import ClusterPyrConv2D as cluster




def ReadFileList(filedir):
    file_list = os.listdir(filedir)
    return  file_list


#Load the image files and partition into blocks
def LoadImageData(fileDir, fileList, dbgDirectory, blockSizeX, blockSizeY ):

    imageBlockSets = []
    imageBlockTargetSets=[]
    count = 0
    for item in fileList:

        #Load the image
        file = fileDir + '/' + item
        image = cv.imread(file, cv.IMREAD_GRAYSCALE)
        size = image.shape

        #Smooth the noisy image to create an 'ideal' image
        imgLevel1 = cv.pyrDown(image,borderType=cv.BORDER_REPLICATE)
        imgLevel2 = cv.pyrDown(imgLevel1, borderType=cv.BORDER_REPLICATE)
        imgReconst1 = cv.pyrUp(imgLevel2, cv.BORDER_REPLICATE)
        imgTarget = cv.pyrUp(imgReconst1, cv.BORDER_REPLICATE)


        #Compute the number of sub-blocks in the image based on sub-block size
        nBlocksX = int(size[1]/blockSizeX)
        nBlocksY = int(size[0]/blockSizeY)

        blockSet = []
        blockTargetSet = []

        #Partition the full size image into X-Y blocks
        for j in range(nBlocksY):
            blockStartY = j*blockSizeY
            blockEndY = blockStartY + blockSizeY
            for i in range(nBlocksX):
                blockStartX = i * blockSizeX
                blockEndX = blockStartX + blockSizeX
                local_block = image[ blockStartY:blockEndY, blockStartX:blockEndX ]
                local_target_block = imgTarget[blockStartY:blockEndY, blockStartX:blockEndX]
                blockSet.append (local_block )
                blockTargetSet.append(local_target_block)

        #Write the block images to disk - for debugging purposes
        blockArray = np.array ( blockSet )
        blockTargetArray = np.array(blockTargetSet)
        outDir = dbgDirectory + '/' + str(count)

        #For debugging purposes, reconstruct the full image from the blocks to compare to the  original input image
        DBG_ImagePartitionTest(blockArray, outDir, image, nBlocksX, nBlocksY, blockSizeX, blockSizeY )
        DBG_ImagePartitionTest(blockTargetArray, outDir, image, nBlocksX, nBlocksY, blockSizeX, blockSizeY)
        imageBlockSets.append(blockArray)
        imageBlockTargetSets.append(blockTargetArray)
        count = count + 1

    imageBlockSetArray = np.array(imageBlockSets)
    imageBlockTargetSetArray = np.array (imageBlockTargetSets)
    return imageBlockSetArray, imageBlockTargetSetArray, size



def DBG_ImagePartitionTest(imageBlockSet, directory, sourceImage, nBlocksX, nBlocksY, blockSizeX, blockSizeY ):

    #check that the directory does not exist, if it does not, create it
    if os.path.isdir(directory) is False:
        os.mkdir ( directory )

    #get the number of blocks
    nBlocks = imageBlockSet.shape[0]

    count = 0
    fileNameRoot = 'DBG_Image_'

    #save each block
    for i in range (nBlocks ):
        fileName = directory + '/' + fileNameRoot + str(count) + '.png'
        cv.imwrite ( fileName, imageBlockSet[i,:,:])
        count = count + 1

    #put the blocks together to reconstruct the image
    nRows = int(nBlocks/nBlocksX)
    nCols = int(nBlocks/nRows)

    reconstructedImage = np.zeros(shape=(nBlocksY*blockSizeY, nBlocksX*blockSizeX), dtype = 'uint16')
    count = 0
    for i in range (nRows ):
        yStart = i * blockSizeY
        yEnd = yStart + blockSizeY

        for j in range ( nCols ):
            xStart = j * blockSizeX
            xEnd = xStart + blockSizeX
            reconstructedImage[ yStart:yEnd, xStart:xEnd ] = imageBlockSet[count,:,:]
            count = count + 1

    reconstructedImageFile = directory + '/' + 'Reconstructed.png'
    sourceImageFile = directory + '/' + 'SourceImage.png'
    cv.imwrite ( reconstructedImageFile, reconstructedImage )
    cv.imwrite ( sourceImageFile, sourceImage )

    return


def ReconstructImage ( imageBlocks, nBlocksX, nBlocksY, blockSX, blockSY  ):
    reconstructedImage = np.zeros(shape=(nBlocksY*blockSX, nBlocksX*blockSX), dtype = 'uint16')

    count = 0
    for i in range (nBlocksY ):
        yStart = i * blockSY
        yEnd = yStart + blockSY

        for j in range ( nBlocksX ):
            xStart = j * blockSX
            xEnd = xStart + blockSX
            reconstructedImage[ yStart:yEnd, xStart:xEnd ] = imageBlocks[count,:,:,0]
            count = count + 1
    return reconstructedImage


#Initialise the neural network
def initParams():
    np.random.seed(11)
    t.random.set_seed(11)
    batch_size = 10
    max_epochs = 8
    filters = 32 #[32, 32, 16]
    return batch_size, max_epochs, filters







#Test programme
def main(filePath, dbgDirectory, modelName, blockSX, blockSY ):


    #Load the data set found in the requested directory
    files = ReadFileList(filePath)


    #Load the image sets and partition into blocks for training
    imageBlockSets, imageBlockTargetSets, imageSize = LoadImageData(filePath, files, dbgDirectory, blockSX, blockSY )
    imageBlockSets = imageBlockSets/65536.0
    imageBlockTargetSets = imageBlockTargetSets/65536.0
    imageBlockSets = imageBlockSets.astype(np.float32)
    imageBlockTargetSets = imageBlockTargetSets.astype(np.float32)

    #Create the model and  train the model
    batchSz, epochs, filters = initParams()
    model =  cluster.Autoencoder(filters, imSX = imageSize[1], imSY = imageSize[0], kX=5, kY=5 )
    model.compile( loss='binary_crossentropy', optimizer='adam' )

    nbr_blocks_total = imageBlockSets.shape[0]*imageBlockSets.shape[1]
    nbr_images = imageBlockSets.shape[0]
    nbr_blocks_per_image = imageBlockSets.shape[1]
    block_count_y = int(imageSize[0]/blockSX)
    block_count_x = int(imageSize[1]/blockSY)

    nbr_trg_blocks = int(0.8*float(nbr_blocks_total))

    imageBlockSetsAggregate = np.reshape(imageBlockSets, newshape=(nbr_images*nbr_blocks_per_image, blockSX, blockSY, 1))
    imageBlockTargetSetsAggregate = np.reshape(imageBlockTargetSets, newshape=(nbr_images*nbr_blocks_per_image, blockSX, blockSY, 1 ))
    trg_set = imageBlockSetsAggregate[0:nbr_trg_blocks,:,:]
    tst_set = imageBlockSetsAggregate[nbr_trg_blocks:nbr_blocks_total]
    trg_set_ref = imageBlockTargetSetsAggregate[0:nbr_trg_blocks, : , : ]
    tst_set_ref = imageBlockTargetSetsAggregate[nbr_trg_blocks:nbr_blocks_total]

#    trg_set = imageBlockSets[0:nbr_trg_blocks,:,:]
#    tst_set = imageBlockSets[nbr_trg_blocks:nbr_blocks_total]
#    trg_set_ref = imageBlockTargetSets[0:nbr_trg_blocks, : , : ]
#    tst_set_ref = imageBlockTargetSets[nbr_trg_blocks:nbr_blocks_total]

    loss = model.fit(trg_set, trg_set_ref, validation_data=(tst_set, tst_set_ref), epochs=epochs, batch_size=batchSz )


    print ( 'loss = ', loss )

    #compute the loss with test data
    plt.plot(range(epochs), loss.history['loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    #Get the overall image size and divide by block sizes
    #to compute number of blocks in x,y directions
    #Then, we know number of blocks  per image to compute the number of images in set
    imageFileRoot = './IMG_RECON_'
    local_count = 0
    for i in range(nbr_images):
        blockIndex = i * nbr_blocks_per_image
        singleImageBlocks = imageBlockSetsAggregate[blockIndex:blockIndex+nbr_blocks_per_image, :, : ]
        singleImageBlocks = singleImageBlocks*65536.0
        imgReconstructed = ReconstructImage(singleImageBlocks, block_count_x, block_count_y, blockSX=blockSX, blockSY=blockSY)
        imageFile = imageFileRoot + str(local_count) + '.tiff'
        cv.imwrite ( imageFile, imgReconstructed)
        local_count = local_count + 1

    #Let's save the model to disk
    _model = model.get_model()
    _model.save(self=_model, filepath=modelName, overwrite = True)

main ( '../PanchromaticImages', '../TestOut1', './Model_2D_ConvAE.hf5', 300, 300)