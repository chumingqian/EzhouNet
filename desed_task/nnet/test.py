


# record_labels = self.dataset.examples[self.dataset.examples_list[idx]]["record_multi_labels"]

record_labels = ['Coarse Crackle', 'Fine Crackle']


# labels = record_labels.split(', ')
labels = []
for label in record_labels:
    if isinstance(label, str):
        labels.extend(label.split(', '))
    else:
        labels.append(label)
