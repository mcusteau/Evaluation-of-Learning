import pandas as pd
import numpy as np

with open('labor-negotiations.data', 'r') as file:
	content = file.readlines()


def readSubTable(start, end, add_label=False, label=None):
	columns_subtable = content[start: start+1][0].replace("\n", "").split(" ")[1:]
	columns_subtable_filtered = list(filter(lambda value: value!='', columns_subtable))
	if(add_label):
		columns_subtable_filtered = columns_subtable_filtered+['label']
	subtable = pd.DataFrame(columns=columns_subtable_filtered)
	row = 0
	for i in range(start+1, end):
		line = content[i:i+1][0].replace("\n", "").split(" ")[1:]
		line_filtered = list(filter(lambda value: value!='', line))
		if(add_label):
			line_filtered = line_filtered + [label]
		subtable.loc[row]= line_filtered
		row+=1
	return subtable

good_events_subtable1 = readSubTable(152, 171)
good_events_subtable2 = readSubTable(173, 192)
good_events_subtable3 = readSubTable(194, 213, add_label=True, label='good')
good_events_train = pd.concat([good_events_subtable1, good_events_subtable2, good_events_subtable3], axis=1)
# replace missing values
good_events_train = good_events_train.replace("*", np.nan)

bad_events_subtable1 = readSubTable(216, 226)
bad_events_subtable2 = readSubTable(228, 238)
bad_events_subtable3 = readSubTable(240, 250, add_label=True, label='bad')
bad_events_train = pd.concat([bad_events_subtable1, bad_events_subtable2, bad_events_subtable3], axis=1)
# replace missing values
bad_events_train = bad_events_train.replace("*", np.nan)

# concatenate both train tables
labour_events_train = pd.concat([good_events_train, bad_events_train], ignore_index=True)


# preprocess test file
with open('labor-neg.test', 'r') as file:
	content = file.readlines()

labour_events_test = pd.DataFrame(columns=list(bad_events_train.columns))
for i in range(len(content)):
	labour_events_test.loc[i] = content[i].replace("\n", "").split(",")
# replace missing values
labour_events_test = labour_events_test.replace("?", np.nan)

#concatenate train and test
labour_events = pd.concat([labour_events_train, labour_events_test], ignore_index=True)

