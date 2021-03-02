import csv


def data_reader(csv_path='even_mnist.csv'):
    """
    Reads CSV file from given path and returns a features matrix and labels vector.
    Args:
        csv_path: Path to the CSV file to be read.
    Returns:
        features: A nested array of the features from the dataset.
        labels: An array of the labels from the dataset.
    """
    features = []
    labels = []

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            row_array = list(map(float, row[0].split(' ')))
            features.append(row_array[:-1])
            labels.append(row_array[-1])

    return features, labels
