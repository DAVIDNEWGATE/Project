#!/usr/bin/python

all_para =['Variable',
 'Variable_1',
 'Variable_2',
 'Variable_3',
 'Variable_4',
 'Variable_5',
 'Variable_6',
 'Variable_7',
 'Variable_8',
 'Variable_9',
 'Variable_10',
 'Variable_11',
 'Variable_12',
 'Variable_13',
 'Variable_14',
 'Variable_15',
 'Variable_16',
 'Variable_17',
 'Variable_18',
 'Variable_19',
 'Variable_20',
 'Variable_21',
 'Variable_22',
 'Variable_23',
 'Variable_24',
 'Variable_25',
 'Variable_26',
 'Variable_27',
 'Variable_28',
 'Variable_29',
 'Variable_30',
 'Variable_31',
 'Variable_32',
 'Variable_33',
 'Variable_34',
 'Variable_35',
 'Variable_36',
 'Variable_37',
 'Variable_38',
 'Variable_39']



exc_para =[#'Variable',
 #'Variable_1',
 'Variable_2',
 'Variable_3',
 'Variable_4',
 'Variable_5',
 'Variable_6',
 'Variable_7',
 'Variable_8',
 'Variable_9',
 'Variable_10',
 'Variable_11',
 'Variable_12',
# 'Variable_13',
 'Variable_14',
 'Variable_15',
 'Variable_16',
 'Variable_17',
 'Variable_18',
 'Variable_19',
 'Variable_20',
 'Variable_21',
 'Variable_22',
 'Variable_23',
 'Variable_24',
 'Variable_25',
# 'Variable_26',
 'Variable_27',
 'Variable_28',
 'Variable_29',
 'Variable_30',
 'Variable_31',
 'Variable_32',
 'Variable_33',
 'Variable_34',
 'Variable_35',
 'Variable_36',
 'Variable_37',
 'Variable_38']
# 'Variable_39']
trans_layer = ['Variable_13','Variable_26']

block_1 = [#'Variable_1',
 'Variable_2',
 'Variable_3',
 'Variable_4',
 'Variable_5',
 'Variable_6',
 'Variable_7',
 'Variable_8',
 'Variable_9',
 'Variable_10',
 'Variable_11',
 'Variable_12',]

block_2 = ['Variable_14',
 'Variable_15',
 'Variable_16',
 'Variable_17',
 'Variable_18',
 'Variable_19',
 'Variable_20',
 'Variable_21',
 'Variable_22',
 'Variable_23',
 'Variable_24',
 'Variable_25',]

block_3 =  ['Variable_27',
 'Variable_28',
 'Variable_29',
 'Variable_30',
 'Variable_31',
 'Variable_32',
 'Variable_33',
 'Variable_34',
 'Variable_35',
 'Variable_36',
 'Variable_37',
 'Variable_38']

fc_layer = ['Variable_39']

prune_para = {}
for k in all_para:
    prune_para[k] = 0.75
prune_para['Variable'] = 0.1
prune_para['Variable_1'] = 0.45
prune_para['Variable_3'] = 0.65

prune_para['Variable_5'] = 0.7
prune_para['Variable_6'] = 0.65

prune_para['Variable_13'] = 0.35
prune_para['Variable_26'] =0.5

prune_para['Variable_34'] = 0.85
prune_para['Variable_35'] = 0.9

prune_para['Variable_36'] = 0.95
prune_para['Variable_37'] = 0.95
prune_para['Variable_38'] = 0.95
prune_para['Variable_39'] = 0.9



dns_para =[#'Variable',
 'Variable_1',
 'Variable_2',
 'Variable_3',
 'Variable_4',
 'Variable_5',
 'Variable_6',
 'Variable_7',
 'Variable_8',
 'Variable_9',
 'Variable_10',
 'Variable_11',
 'Variable_12',
 'Variable_13',
 'Variable_14',
 'Variable_15',
 'Variable_16',
 'Variable_17',
 'Variable_18',
 'Variable_19',
 'Variable_20',
 'Variable_21',
 'Variable_22',
 'Variable_23',
 'Variable_24',
 'Variable_25',
 'Variable_26',
 'Variable_27',
 'Variable_28',
 'Variable_29',
 'Variable_30',
 'Variable_31',
 'Variable_32',
 'Variable_33',
 'Variable_34',
 'Variable_35',
 'Variable_36',
 'Variable_37',
 'Variable_38',
 'Variable_39']

crate = {}
for k in all_para:
    crate[k] = 3
crate['Variable'] = 0
crate['Variable_1'] = 1
crate['Variable_13'] = 1
crate['Variable_26'] =1

inqpercen_para = {}
for k in all_para:
    inqpercen_para[k] = 1.0
    
inq_para = {}
for k in all_para:
    inq_para[k] = 16
    
#inq_para['Variable'] = 256
#inq_para['Variable_1'] = 128
#inq_para['Variable_13'] = 128
#inq_para['Variable_26'] =128

inqprune_para = {}
for k in all_para:
    inqprune_para[k] = 1-0.75
inqprune_para['Variable'] = 1-0.1
inqprune_para['Variable_1'] = 1-0.45
inqprune_para['Variable_3'] = 1-0.65

inqprune_para['Variable_5'] = 1-0.7
inqprune_para['Variable_6'] = 1-0.65

inqprune_para['Variable_13'] = 1-0.35
inqprune_para['Variable_26'] =1-0.5

inqprune_para['Variable_34'] = 1-0.85
inqprune_para['Variable_35'] = 1-0.9

inqprune_para['Variable_36'] = 1-0.95
inqprune_para['Variable_37'] = 1-0.95
inqprune_para['Variable_38'] = 1-0.95
inqprune_para['Variable_39'] = 1-0.9    


kmeans_para = {}
for k in all_para:
    kmeans_para[k] = 64

len_dict = {'Variable': 432,
 'Variable_1': 1728,
 'Variable_10': 13392,
 'Variable_11': 14688,
 'Variable_12': 15984,
 'Variable_13': 25600,
 'Variable_14': 17280,
 'Variable_15': 18576,
 'Variable_16': 19872,
 'Variable_17': 21168,
 'Variable_18': 22464,
 'Variable_19': 23760,
 'Variable_2': 3024,
 'Variable_20': 25056,
 'Variable_21': 26352,
 'Variable_22': 27648,
 'Variable_23': 28944,
 'Variable_24': 30240,
 'Variable_25': 31536,
 'Variable_26': 92416,
 'Variable_27': 32832,
 'Variable_28': 34128,
 'Variable_29': 35424,
 'Variable_3': 4320,
 'Variable_30': 36720,
 'Variable_31': 38016,
 'Variable_32': 39312,
 'Variable_33': 40608,
 'Variable_34': 41904,
 'Variable_35': 43200,
 'Variable_36': 44496,
 'Variable_37': 45792,
 'Variable_38': 47088,
 'Variable_39': 4480,
 'Variable_4': 5616,
 'Variable_5': 6912,
 'Variable_6': 8208,
 'Variable_7': 9504,
 'Variable_8': 10800,
 'Variable_9': 12096}
    
