from HodaDatasetReader import read_hoda_dataset
import pickle

train_images, train_labels = read_hoda_dataset('./DigitDB/Train 60000.cdb', reshape=False)
test_images, test_labels = read_hoda_dataset('./DigitDB/Test 20000.cdb', reshape=False)
remaining_images, remaining_labels = read_hoda_dataset('./DigitDB/RemainingSamples.cdb', reshape=False)
# because of the dataset, it's better to shuffle the dataset to increase accuracy and avoid the network from memorising
train_images, train_labels = shuffle(np.array(train_images), np.array(train_labels))
test_images, test_labels = shuffle(np.array(test_images), np.array(test_labels))
remaining_images, remaining_labels = shuffle(np.array(remaining_images), np.array(remaining_labels))

# In order to save dataset to pickle
listNames = ['train_images', 'train_labels', 'test_images', 'test_labels', 'remaining_images', 'remaining_labels']
for i in listNames:
    pickle_out = open("DigitDB/{}.pickle".format(i), 'wb')
    pickle.dump(i, pickle_out)
    pickle_out.close()

# Load the dataset after saving it
pickle_in = open("DigitDB/train_images.pickle", "rb")
train_images = pickle.load(pickle_in)
