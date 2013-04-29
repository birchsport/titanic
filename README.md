# titanic #

Example code for solving the Titanic prediction problem (https://www.kaggle.com/c/titanic-gettingStarted) found on Kaggle.  This example uses the Weka Data Mining Libraries to perform our classifications and predictions. Note, we are using Weka version 3.6.9.

## Data Cleanup/Initialization ##

Before we begin, we have to clean up the data files provided by Kaggle (these cleanup steps have already been performed on the committed files).  The first step is to remove the nested '""' (quotation marks) from the files.  This was simply a straight search and replace operation in my editor.

The next step is to convert the CSV formatted files into the ARFF format.  The ARFF format provides more detailed information about the type of data in the CSV files.  To perform this conversion, you can use the CSVLoader from the Weka libraries.

```
java -cp lib/weka.jar weka.core.converters.CSVLoader test.csv > test.arff
java -cp lib/weka.jar weka.core.converters.CSVLoader train.csv > train.arff
```

Once we have created the ARFF files, we need to clean them up a little bit.  First, we identify any 'string' column to be of type string, and not nominal.  Then we ensure that nominal values are in the same order for both files (VERY IMPORTANT!).  Here is what the header section of the ARFF file should look like:

```
@attribute survived {0,1}
@attribute pclass numeric
@attribute name string
@attribute sex {male,female}
@attribute age numeric
@attribute sibsp numeric
@attribute parch numeric
@attribute ticket string
@attribute fare numeric
@attribute cabin string
@attribute embarked {Q,S,C}
```

## Training, Predicting, and Verifying the data ##

Now that we have cleaned up our data, we are ready to run the code.  I have included the Eclipse project files to make it easy for anyone to import this project into Eclipse and go.  I have also included an Ant build file to compile and run everything as well.  If you don't have either of those options, you are on your own.

### Training ###

To train the classifier, execute the 'titanic.weka.Train' class or run 'ant train' in a terminal.  This will load the training data, create and train a Classifier, and write the Classifier to disk.

### Predicting ###

To create a prediction, execute the 'titanic.weka.Predict' class or run 'ant predict'.  This will load the test data, read the trained Classifier from disk, and produce a 'predict.csv'.  This CSV file is in a suitable format to submit to Kaggle.

### Verifying ###

To verify our predictions, execute the 'titanic.weka.Verify' class or run 'ant verify'.  This will load our prediction results, read the trained Classifier from disk, then evaluate the classification performance.  You will see output similar to this:

```
Correctly Classified Instances         418              100      %
Incorrectly Classified Instances         0                0      %
Kappa statistic                          1     
Mean absolute error                      0.1409
Root mean squared error                  0.1986
Relative absolute error                 30.3515 %
Root relative squared error             41.2246 %
Total Number of Instances              418     
```
