shuf dataset.csv > dataset.csv.tmp
head -15000 dataset.csv.tmp > train_data_valid.csv
tail -5695 dataset.csv.tmp > testing_data.csv
rm dataset.csv.tmp