Pandas is used to read in the TSV files.
pip install pandas should correctly download the library.

NLTK is used to implement stoplisting and lemmatisation during preprocessing.
I have included the check_nltk_resources function in line 58-74 to download required resources if necessary.
pip install nltk should correctly download the library.

Seaborn and matplotlib are used to generate the confusion matrix graph.
pip install seaborn and pip install matplotlib should correctly download the libraries.

The file_name in the save_results functions are predefined.
If you are going to save new files, i recommend renaming the file_name in the function to prevent overwriting tsv files I submitted.
The function runs normally I am just not sure what you expect for the output_files option since the instructions are unclear.