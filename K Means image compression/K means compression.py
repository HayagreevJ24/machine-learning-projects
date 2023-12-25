# An Application of K means clustering - Image compression
import numpy as np
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits import mplot3d
from tqdm import tqdm

"""
1. Class KMeansCompressor:
    (a) Constructor Method:
    Args: Image: numpy.ndarray of shape (rows, cols, 3)
          NumberOfColors: int any number from 16777216 (number of cluster centers in K means)
          Epochs: int: Number of times K means needs to be run
    Instance variables apart from constructed arguments:
          self.originalDim: tuple of shape (rows, cols, 3) - Original image dimensions from Image
          self.Compressed: Bool - False (indicates whether the image has been compressed)
          self.Kmeansstatus: bool - False (indicates whether K means has been run)
          self.FlattenedImage: numpy.ndarray of shape (rows*cols, 3) - Flattened image
          self.CompressedImage: numpy.ndarray of shape (rows, cols, 3) - Compressed image
          self.clusterCenters: numpy.ndarray of shape (NumberOfColors, 3) - Cluster centers
          self.clusterAssignments: numpy.ndarray of shape (rows*cols, 1) - Cluster assignments for each pixel. Values from 0 to NumberOfColors-1
          self.PSNR: float - Peak signal-to-noise ratio
          self.MSE: float - Mean squared error
          self.CompressionRatio: float - Compression ratio
          
    (a) Method 1: flattenImage - Flattens the image
    Args: self
    Returns: None

    (b) Method 2: [Part of K means routine] - assignClusterCenters - Assigns cluster centers to each pixel
    Args: self
    Returns: None

    (c) Method 3: [Part of K means routine] - updateClusterCenters - Updates cluster centers based on cluster assignments
    Args: self
    Returns: None

    (d) Method 4: [Part of K means routine] - runKmeans - Runs K means for Epochs number of times
    Args: self
    Returns: None

    (e) Method 5: UnflattenImage - Unflattens the image
    Args: self, imageToUnflatten: numpy.ndarray of shape (rows*cols, 3)
    Returns: numpy.ndarray of shape (rows, cols, 3) from self.originalDim

    (f) Method 6: compress - Compresses the image
    Args: self
    Returns: None
    
    (g) Method 7: [Part of visualisation subroutine] - visualiseClusterCenters - Visualises the cluster centers with all the image colors
    Args: self
    Returns: None
    
    (h) Method 8: [Part of visualisation subroutine] - showimage - Either shows the original image or the compressed image or both
    Args: self.Image
    Returns: None
    
    (i) Method 9: GetMetrics - Prints the calculated PSNR, MSE and compression ratio
    Args: self
    Returns: None
    
    - Note that the algorithm for each method has not been described but appropriate checks are made and exceptions are raised if necessary. 
    - Note that all methods are by reference and modify the instance variables directly. They do not return anything. 
"""


class DimensionError(Exception):
    pass


class KMeansError(Exception):
    pass


class NotCompressedError(Exception):
    pass


class KMeansCompressor:
    def __init__(self, Image, NumberOfColors, Epochs):
        self.Image = Image
        self.NumberOfColors = NumberOfColors
        self.Epochs = Epochs
        self.originalDim = Image.shape
        self.Compressed = False
        self.Kmeansstatus = False
        self.FlattenedImage = None
        self.CompressedImage = None  # Not initialised yet - will be initialised in compress method.
        self.clusterCenters = np.random.randint(0, 256, size=(self.NumberOfColors, 3))
        self.clusterAssignments = None # Not initialised yet - will be initialised in assignClusterCenters method.
        self.PSNR = None
        self.MSE = None
        self.CompressionRatio = None

    def flattenImage(self) -> None:
        print(f'\n\nOriginal image dimension:  {self.originalDim}')

        if self.originalDim[2] == 4:
            raise DimensionError('Image has alpha channel and cannot be compressed with K means.')

        totalPixelCount = self.Image.shape[0] * self.Image.shape[1]
        self.FlattenedImage = self.Image.reshape(totalPixelCount, 3)

        print(f'Flattened image dimension: {self.FlattenedImage.shape}')
        print(f'Flattened image successfully.')

    def assignClusterCenters(self, epochs) -> None:
        # print(f"\nAssigning cluster centers for epoch {epochs + 1}: ")
        # Uncomment above line when using non vectorised approach.
        # VECTORISED IMPLEMENTATION
        distancesToCenters = np.linalg.norm(self.FlattenedImage[:, np.newaxis, :] - self.clusterCenters, axis=2)
        # Above line takes image of shape (r * c, 3) and makes it (r * c, 1, 3) and calculates the distance by broadcasting it into cluster centers of shape (NumberOfColors, 3). The resulting array is (r * c, NumberOfColors) and contains the distances of each pixel to each cluster center.
        # Then, argmin below takes the array of shape (r * c, NumberOfColors) and returns an array of shape (numberOfPixels, 1) containing the index of the minimum distance cluster center for each pixel. axis = 1 is important as min is found over columns (clusters) for each pixel.
        self.clusterAssignments = np.argmin(distancesToCenters, axis=1)

        # NON VECTORISED IMPLEMENTATION
        # for pixelNum in tqdm(range(self.originalDim[0] * self.originalDim[1])):
        #     pixel = self.FlattenedImage[pixelNum]
        #     distancesToCenters = np.zeros(self.NumberOfColors)
        #     for centerNum in range(self.NumberOfColors):
        #         distancesToCenters[centerNum] = np.linalg.norm(pixel - self.clusterCenters[centerNum])
        #     self.clusterAssignments[pixelNum] = np.argmin(distancesToCenters)

    def updateClusterCenters(self, epochs) -> None:
        # print(f"\nUpdating cluster centers for epoch {epochs + 1}: ")
        # Uncomment above line when using non vectorised approach.

        # VECTORISED IMPLEMENTATION
        newclusterCenters = np.zeros((self.NumberOfColors, 3))
        assignmentCounts = np.zeros(self.NumberOfColors)

        # Iterate over each channel and accumulate weighted sums
        for channel in range(3):
            channel_weights = self.FlattenedImage[:, channel]
            newclusterCenters[:, channel] += np.bincount(self.clusterAssignments, weights=channel_weights,
                                                         minlength=self.NumberOfColors)

        # Count the number of assignments for each cluster
        assignmentCounts += np.bincount(self.clusterAssignments, minlength=self.NumberOfColors)

        # Divide by assignment counts to get the mean position, ignoring zero counts to avoid division by zero
        nonEmptyClusters = np.where(assignmentCounts != 0)[0]
        newclusterCenters[nonEmptyClusters] /= assignmentCounts[nonEmptyClusters, None]

        self.clusterCenters = newclusterCenters
        # Only assigns the non-empty clusters --> Leaves the rest alone to avoid division by zero error.

        # NON VECTORISED IMPLEMENTATION
        # for centerNum in tqdm(range(self.NumberOfColors)):
        #     assignmentCount = 0
        #     newCenter = np.zeros(3)
        #     for pixels in range(self.originalDim[0] * self.originalDim[1]):
        #         if self.clusterAssignments[pixels] == centerNum:
        #             newCenter += self.FlattenedImage[pixels]
        #             assignmentCount += 1
        #     try:
        #         newCenter /= assignmentCount
        #     except ZeroDivisionError:
        #         pass  # Ignore this center and let it continue having its previous initialisation since no pixels were assigned to it this time.
        #
        #     self.clusterCenters[centerNum] = newCenter

    def runKmeans(self) -> None:
        if self.FlattenedImage is None:
            raise DimensionError('Please flatten the image with self.flattenImage() before running K means')
        elif self.Kmeansstatus:
            warnings.warn('K means has already been run. Running it again will overwrite the previous cluster centers and cluster assignments')
        else:
            print(f'\n\nRunning K means for {self.Epochs} epochs: ')
            for epoch in tqdm(range(self.Epochs)):
                # print(f'\n\nEpoch {epoch + 1} of {self.Epochs}....')
                # Uncomment above line when using non vectorised approach.
                self.assignClusterCenters(epoch)
                self.updateClusterCenters(epoch)
            self.Kmeansstatus = True

    def UnflattenImage(self, imageToUnflatten) -> np.ndarray:
        print(f'\nFlattened image dimension: {imageToUnflatten.shape}')
        print(f'Unflattened image dimension: {self.originalDim}')
        print(f'Unflattened image successfully.')
        return imageToUnflatten.reshape(self.originalDim)

    def compress(self) -> None:  # Expect K means to have taken place first before compression
        if not self.Kmeansstatus:
            raise KMeansError('Please run K means first before compressing the image')
        elif self.Compressed:
            warnings.warn(
                'Image has already been compressed. Compressing again will overwrite the previous compressed image')
        else:
            print(f'\n\nCompressing image with results from K means: ')
            self.CompressedImage = np.zeros((self.originalDim[0] * self.originalDim[1], 3))
            for pixelNum in tqdm(range(self.originalDim[0] * self.originalDim[1])):
                self.CompressedImage[pixelNum] = self.clusterCenters[int(self.clusterAssignments[pixelNum])]
            # To ensure that pixel values are in the range 0 to 1
            self.CompressedImage = self.CompressedImage / 255

            self.CompressedImage = self.UnflattenImage(self.CompressedImage)
            self.Compressed = True

            writeToFile = input('Do you want to write the compressed image to a file? (y/n): ').lower()
            match writeToFile:
                case 'y':
                    plt.imsave('CompressedImage.png', self.CompressedImage)
                    print('Image successfully written to file. \'CompressedImage.png\'.')
                case 'n':
                    pass
                case _:
                    raise ValueError('Invalid option! Please enter either "y" or "n"')

    def visualiseColors(self) -> None:
        if not self.Kmeansstatus:
            raise KMeansError('Please run K means first before visualising the cluster centers')
        else:
            # 3D plot of image colours and cluster centers
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection='3d')
            ax.set_xlabel('Red')
            ax.set_ylabel('Green')
            ax.set_zlabel('Blue')
            ax.set_title('Image color space and cluster centers')
            ax.scatter3D(self.FlattenedImage[:, 0], self.FlattenedImage[:, 1], self.FlattenedImage[:, 2], c=self.FlattenedImage/255, cmap='viridis', s=10, marker='.')
            ax.scatter3D(self.clusterCenters[:, 0], self.clusterCenters[:, 1], self.clusterCenters[:, 2], cmap='viridis', c='black', s=100, marker='x')
            plt.show()
            plt.figure(figsize=(10, 10))
            plt.title('Cluster centers')
            plt.imshow(self.clusterCenters.reshape(self.NumberOfColors, 1, 3).astype(np.uint8))
            plt.show()

    def showimage(self, imageToShow) -> None:
        match imageToShow:
            case 'original':
                plt.figure(figsize=(10, 10))
                plt.imshow(self.Image)
                plt.show()
            case 'compressed':
                if not self.Compressed:
                    raise NotCompressedError('Please compress the image first before showing the compressed image')
                else:
                    plt.figure(figsize=(10, 10))
                    plt.imshow(self.CompressedImage)
                    plt.show()
            case 'both':
                if not self.Compressed:
                    raise NotCompressedError('Please compress the image first before showing the compressed image')
                else:
                    plt.figure(figsize=(10, 10))
                    plt.subplot(1, 2, 1)
                    plt.imshow(self.Image)
                    plt.subplot(1, 2, 2)
                    plt.imshow(self.CompressedImage)
                    plt.show()
            case _:
                raise ValueError(
                    'Please enter either "original" or "compressed" or "both" to show the respective images')

    def GetMetrics(self) -> None:
        if not self.Compressed:
            raise KMeansError('Please compress the image first before getting metrics')

        self.PSNR = 10 * np.log10(255 ** 2 / (np.mean((self.Image - self.CompressedImage) ** 2)))
        self.MSE = np.mean((self.Image - self.CompressedImage) ** 2)
        self.CompressionRatio = (2**24) / (2**self.NumberOfColors)

        print(f'\nPeak signal-to-noise ratio: {self.PSNR}')
        print(f'Mean squared error: {self.MSE}')
        print(f'Compression ratio: {self.CompressionRatio}')

        writemetrics = input('Do you want to write the metrics to a file? (y/n): ').lower()
        match writemetrics:
            case 'y':
                with open('Metrics.txt', 'w') as f:
                    f.write(f'Peak signal-to-noise ratio: {self.PSNR}\n')
                    f.write(f'Mean squared error: {self.MSE}\n')
                    f.write(f'Compression ratio: {self.CompressionRatio}\n')
                print('Metrics successfully written to file. \'Metrics.txt\'.')
            case 'n':
                pass
            case _:
                raise ValueError('Invalid option! Please enter either "y" or "n"')



# ENDCLASS

# Testing the program
myCompressor = KMeansCompressor(plt.imread('beautiful-nature-mountain-scenery-with-flowers-free-photo.webp'), 10, 100)

myCompressor.flattenImage()
myCompressor.runKmeans()
myCompressor.compress()
myCompressor.GetMetrics()
myCompressor.showimage('both')
myCompressor.visualiseColors()

# END
