#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-

import sys
import math
import json

# todo: incorporate timeouts and inconsistent solution status in general

# This script compares several SoPlex .json files.
# The first argument is the default run that is compared to all other runs.
# Set values to be compared and respective shift values in arrays 'compareValues' and 'shift'

def printUsage(name):
    print('compare several runs of the same testset')
    print('usage: '+name+' [prf=<performance-profile>.prf] [ignore=<instance1>,<instance2>,...] <soplex_test_run1>.json [<soplex_test_run2>.json ...]')
    quit()

# compute speed-up or slow-down factors for all instances of two settings
# compareValue is the name of the value of the loaded dictionary
# compareName is the new name to be used as storage for the factors
def computeFactor(compareValue, defaultSetting, compareSetting):
    "compute speed-up or slow-down factors for all instances of two settings"
    for instance in results[defaultSetting]:
        if not instance in results[compareSetting]:
            factors[compareValue][compareSetting][instance] = 1.0
        elif not compareValue in results[compareSetting][instance]:
            factors[compareValue][compareSetting][instance] = 1.0
        else:
          value = results[compareSetting][instance][compareValue]
          defvalue = results[defaultSetting][instance][compareValue]
          if defaultSetting == compareSetting:
              factors[compareValue][compareSetting][instance] = defvalue
          elif defvalue == 0:
              factors[compareValue][compareSetting][instance] = 1.0
          else:
              factors[compareValue][compareSetting][instance] = float(value) / float(defvalue)

# update the current (shifted) geometric metric
def updateGeoMean(new, mean, count, shift):
    assert mean > 0
    return math.pow(float(mean), float(count-1)/float(count)) * math.pow(float(new)+shift, 1.0/float(count))

# print header lines
def printHeader():
    border = '-'*(namelength) + '+' + '-'*(sum(length)+1) + '+'
    output1 = '-'.join([settings[0],githash[0]]).rjust(namelength+sum(length)+1)
    output2 = 'name'.ljust(namelength) + ' '
    # print name of compareValue of first setting
    for i,c in enumerate(compareValues):
        output2 += c.rjust(length[i])
    # print compareValues and factornames for other settings
    for i,s in enumerate(settings[1:]):
        output1 += ' |' + '-'.join([s,githash[i+1]]).center(sum(length) + len(compareValues)*factorlength)
        border += '-'*(sum(length) + len(compareValues)*factorlength + 1) + '+'
        output2 += ' |'
        for ic,c in enumerate(compareValues):
            output2 += c.rjust(length[ic])
        for c in compareValues:
            output2 += (c+'Q').rjust(factorlength)
    print(border)
    print(output1)
    print(output2)
    print(border)

# print usage and exit
if len(sys.argv) < 2:
    printUsage(sys.argv[0])

# set compare values, used to identify the values from computeFactor
compareValues = ['solvetime','iters']
shift = [0.1, 10, 1]

# look for instances to ignore and check whether performance profile files should be created
ignore = []
performanceProFile = ''
for a in sys.argv[1:]:
    if a.startswith('ignore='):
        ignore = a[7:].split(',')
        sys.argv.remove(a)
    elif a.startswith('prf='):
        performanceProFile = a[4:]
        sys.argv.remove(a)

# parse testset from first file
testset = sys.argv[1].split('/')[-1].split('.')[1]

runs = len(sys.argv)
results = {}
factors = {}
settings = []
version = []
opt = []
githash = []

# load given .json files
for run in range(1,runs):
    dataname = sys.argv[run]
    setting = sys.argv[run].split('/')[-1].split('.')[-2]
    v = '.'.join(sys.argv[run].split('/')[-1].split('.')[2:-7])[7:]
    version.append(v)
    setting = '.'.join([v,setting])
    if setting in settings:
        setting = '_'.join([setting, str(run)])
    settings.append(setting)
    opt.append(sys.argv[run].split('/')[-1].split('.')[-3])
    # check for identical testset
    if not testset == dataname.split('/')[-1].split('.')[1]:
        print('inconsistent testsets')
        quit()
    with open(dataname) as f:
        results[setting] = json.load(f)

# set default setting to compare with
default = settings[0]

# extract instance names
instances = list(results[default].keys())
namelength = 16
for i in instances:
    namelength = max(len(i), namelength)

# extract git hashes
for s in settings:
    githash.append(results[s][instances[0]]['githash'])

# check all settings for aborts or instances to ignore and remove them
aborts = ''
mintime = 0
for s in settings:
    for i in instances:
        if i not in results[s]:
            aborts = aborts + i + '\n'
            instances.remove(i)
            del results[default][i]
        elif results[s][i]['status'] == 'abort' or i in ignore:
            aborts = aborts + i + '\n'
            instances.remove(i)
            del results[s][i]
        else:
            results[s][i]['solvetime'] = max( mintime, results[s][i]['solvetime'] )

# compute all the comparison factors
for c in compareValues:
    factors[c] = {}
    for s in settings:
        factors[c][s] = {}
        computeFactor(c, default, s)

# compute column lengths
namelength += 1
length = []
for idx,c in enumerate(compareValues):
    length.append(len(c))
    for s in settings:
        for i in instances:
            length[idx] = max(length[idx],len(str(results[s][i][c])))
    length[idx] += 2

factorlength = max(length)

printHeader()

count = 0

sumValue = {}
meanValue = {}
shmeanValue = {}

#initialize mean and shmean variables
for c in compareValues:
    sumValue[c] = {}
    meanValue[c] = {}
    shmeanValue[c] = {}
    for s in settings:
        sumValue[c][s] = 0
        meanValue[c][s] = 1.0
        shmeanValue[c][s] = 1.0

# print data for all instances with the computed length
for i in sorted(instances):
    count += 1
    output = i.ljust(namelength)
    for ic,c in enumerate(compareValues):
        # print results of default settings
        value = results[default][i][c]
        sumValue[c][default] += value
        meanValue[c][default] = updateGeoMean(value, meanValue[c][default], count, 0.0001)
        shmeanValue[c][default] = updateGeoMean(value, shmeanValue[c][default], count, shift[ic])
        output += str(value).rjust(length[ic])

    # print results of remaining settings
    for ids, s in enumerate(settings[1:]):
        output += '  '
        for ic,c in enumerate(compareValues):
            value = results[s][i][c]
            sumValue[c][s] += value
            meanValue[c][s] = updateGeoMean(value, meanValue[c][s], count, 0.0001)
            shmeanValue[c][s] = updateGeoMean(value, shmeanValue[c][s], count, shift[ic])
            output += str(value).rjust(length[ic])
        # print calculated factors
        for ic,c in enumerate(compareValues):
            output += '{0:{width}.2f}'.format(factors[c][s][i], width=factorlength)

    print(output)

shmeanValue[c][default] -= shift[ic]
shmeanValue[c][s] -= shift[ic]


printHeader()

# print summary of comparision
output1 = 'sum:'.ljust(namelength)
output2 = 'geo mean:'.ljust(namelength)
output3 = 'shifted:'.ljust(namelength)
# print values of default setting
for ic,c in enumerate(compareValues):
    output1 += str(round(sumValue[c][default],2)).rjust(length[ic])
    output2 += str(round(meanValue[c][default],2)).rjust(length[ic])
    output3 += str(round(shmeanValue[c][default],2)).rjust(length[ic])
    # padding to next setting
    output1 += '  '
    output2 += '  '
    output3 += '  '
for ids, s in enumerate(settings[1:]):
    # values of other settings
    for ic,c in enumerate(compareValues):
        output1 += str(round(sumValue[c][s],2)).rjust(length[ic])
        output2 += str(round(meanValue[c][s],2)).rjust(length[ic])
        output3 += str(round(shmeanValue[c][s],2)).rjust(length[ic])
        # padding to next setting
        output1 += '  '
        output2 += '  '
        output3 += '  '
    # padding to next setting
    output1 += ' '*(len(compareValues)*factorlength + 2)
    output2 += ' '*(len(compareValues)*factorlength + 2)
    output3 += ' '*(len(compareValues)*factorlength + 2)
print(output1)
print(output2)
print(output3)

# print aborted and ignored instances
if not aborts == '':
    print('\naborted and ignored instances:')
    print(aborts)

# generate performance profile files
if not performanceProFile == '':
    for idx,s in enumerate(settings):
        filename = performanceProFile+'.'+testset+'.soplex-'+s+'.prf'
        with open(filename, 'w') as f:
            f.write('---\n')
            f.write('algname: '+s+'\n')
            f.write('success: optimal\n')
            f.write('free_format: True\n')
            f.write('---\n')
            for i in sorted(instances):
                if results[s][i]['solvetime'] < 1.0:
                    time = 1.0
                else:
                    time = results[s][i]['solvetime']
                status = results[s][i]['status'].replace(' ', '_')
                output = i.ljust(22)
                output += status.rjust(18)
                output += str(time).rjust(14)
                output += '\n'
                f.write(output)
