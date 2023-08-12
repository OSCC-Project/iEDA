#! /usr/bin/env python3

import sys
import math
import json
import fractions

# this function checks for the type of parsed string
def typeofvalue(text):
    try:
        int(text)
        return int
    except ValueError:
        pass

    try:
        float(text)
        return float
    except ValueError:
        pass

    try:
        fractions.Fraction(text)
        return fractions.Fraction
    except ValueError:
        pass

    return str

def fracttofloat(value):
    try:
        float(value)
        return float(value)
    except ValueError:
        return float(fractions.Fraction(value))

if len(sys.argv) > 2:
    if not len(sys.argv) == 3:
        print('usage (for decomposition solve output): '+sys.argv[0]+'-d <soplex_test_run>.out')
        quit()
elif not len(sys.argv) == 2:
    print('usage: '+sys.argv[0]+' <soplex_test_run>.out')
    quit()

# if the -d flag is provided, then we must produce the decomposition solve output
decomp = False
if sys.argv[1] == '-d':
    decomp = True

# specify columns for the output (can be modified)
columns = ['name','rows','cols','pviol','dviol','iters','polish','refs','solvetime','value','status']
ncolumns = len(columns)

if decomp:
    decompcolumns = ['ppiv','dpiv','avgpdegen','avgddegen','redprob','compprob','rediter','compiter','algiter']
    columns.extend(decompcolumns)

if decomp:
    outname = sys.argv[2]
else:
    outname = sys.argv[1]
dataname = outname.replace('.out','.json')

outfile = open(outname,'r')
outlines = outfile.readlines()
outfile.close()

testset = outname.split('/')[-1].split('.')[1]
testname = 'testset/'+testset+'.test'
soluname = 'testset/'+testset+'.solu'

settings = outname.split('/')[-1].split('.')[-2]

# maximum length of instance names
namelength = 18

# tolerance for solution value check
tolerance = 1e-6

instances = {}
stats = False
printedIdentifier = False
section = 'soplex'

for idx, outline in enumerate(outlines):
    # print identifier
    if outline.startswith('@01'):
        # convert line to used instance name
        linesplit = outline.split('/')
        linesplit = linesplit[-1].rstrip(' \n').rstrip('.gz').rstrip('.GZ').rstrip('.z').rstrip('.Z')
        linesplit = linesplit.split('.')
        instancename = linesplit[0]
        for i in range(1, len(linesplit)-1):
            instancename = instancename + '.' + linesplit[i]
        length = len(instancename)
        if length > namelength:
            shortname = instancename[0:int(namelength/2)-1] + '~' + instancename[length-int(namelength/2):]
        else:
            shortname = instancename

        # initialize new data set
        instances[instancename] = {}
        instances[instancename]['status'] = 'abort'
        instances[instancename]['name'] = shortname
        instances[instancename]['settings'] = settings
        instances[instancename]['testset'] = testset
        # wait for statistics block
        stats = False

    # invalidate instancename
    elif outline.startswith('=ready='):
        section = 'soplex'
        instancename = ''

    elif outline.startswith('=perplex='):
        section = 'perplex'
        stats = False
        instances[instancename]['perplex'] = 'unknown'

    elif outline.startswith('=qsoptex='):
        section = 'qsoptex'
        stats = False
        instances[instancename]['qso:stat'] = 'unknown'
        instances[instancename]['qso:lpval'] = '--'
        instances[instancename]['qso:time'] = instances[instancename]['timelimit']
        instances[instancename]['qso:prec'] = 64

    elif section == 'perplex':
        if outline.find('No such file or directory') >= 0:
            instances[instancename]['perplex'] = 'readerror'
        elif outline.startswith('Basis read'):
            instances[instancename]['perplex'] = 'timeout'
        elif outline.startswith('Solution is optimal'):
            if instances[instancename]['perplex'] == 'dinfeas':
                instances[instancename]['perplex'] = 'optimal'
            elif instances[instancename]['perplex'] == 'pdinfeas' or instances[instancename]['perplex'] == 'timeout':
                instances[instancename]['perplex'] = 'pinfeas'
        elif outline.startswith('Solution is feasible'):
            if instances[instancename]['perplex'] == 'pinfeas':
                instances[instancename]['perplex'] = 'optimal'
            elif instances[instancename]['perplex'] == 'pdinfeas' or instances[instancename]['perplex'] == 'timeout':
                instances[instancename]['perplex'] = 'dinfeas'
        elif outline.startswith('Solution is not ') and instances[instancename]['perplex'] == 'timeout':
            instances[instancename]['perplex'] = 'pdinfeas'

    elif section == 'qsoptex':
        if outline.startswith('Time for SOLVER:'):
            instances[instancename]['qso:time'] = fracttofloat(outline.split()[-2])
        elif outline.find('Problem Solved Exactly') >= 0:
            instances[instancename]['qso:stat'] = 'optimal'
        elif outline.find('Problem Is Infeasible') >= 0:
            instances[instancename]['qso:stat'] = 'infeasible'
        elif outline.startswith('@24') and instances[instancename]['qso:time'] >= instances[instancename]['timelimit']:
            instances[instancename]['qso:stat'] = 'timeout'
        elif outline.startswith('LP Value'):
            instances[instancename]['qso:lpval'] = outline.split()[2].rstrip(',')
        elif outline.find('Trying mpf with') >= 0:
            instances[instancename]['qso:prec'] = max( int(outline.split()[3]), instances[instancename]['qso:prec'] )

    elif outline.startswith('SoPlex version'):
        instances[instancename]['githash'] = outline.split()[-1].rstrip(']')[0:9]
        if not printedIdentifier:
            printedIdentifier = True
            print('\n')
            print(outline)

    elif outline.startswith('Primal solution infeasible') or outline.startswith('Dual solution infeasible'):
        instances[instancename]['status'] = 'fail'

    elif outline.startswith('real:timelimit'):
        instances[instancename]['timelimit'] = fracttofloat(outline.split()[-1])

    elif outline.startswith('Statistics'):
        stats = True

    if stats:
        if outline.startswith('SoPlex status'):
            if outline.find('time limit') >= 0:
                instances[instancename]['status'] = 'timeout'
            else:
                checkstat = instances[instancename]['status']
                reportstat = outline.split()[-1].strip('[]')
                if reportstat == 'optimal' and checkstat == 'fail':
                    instances[instancename]['status'] = checkstat
                else:
                    instances[instancename]['status'] = reportstat

        elif outline.startswith('Solution'):
            instances[instancename]['value'] = fracttofloat(outlines[idx+1].split()[-1])

        elif outline.startswith('Original problem'):
            instances[instancename]['cols'] = int(outlines[idx+1].split()[2])
            instances[instancename]['boxedcols'] = int(outlines[idx+2].split()[2])
            instances[instancename]['lbcols'] = int(outlines[idx+3].split()[3])
            instances[instancename]['ubcols'] = int(outlines[idx+4].split()[3])
            instances[instancename]['freecols'] = int(outlines[idx+5].split()[2])
            instances[instancename]['rows'] = int(outlines[idx+6].split()[2])
            instances[instancename]['equalrows'] = int(outlines[idx+7].split()[2])
            instances[instancename]['rangedrows'] = int(outlines[idx+8].split()[2])
            instances[instancename]['lhsrows'] = int(outlines[idx+9].split()[2])
            instances[instancename]['rhsrows'] = int(outlines[idx+10].split()[2])
            instances[instancename]['freerows'] = int(outlines[idx+11].split()[2])
            instances[instancename]['nonzeros'] = int(outlines[idx+12].split()[2])
            instances[instancename]['colnonzeros'] = fracttofloat(outlines[idx+13].split()[3])
            instances[instancename]['rownonzeros'] = fracttofloat(outlines[idx+14].split()[3])
            instances[instancename]['sparsity'] = fracttofloat(outlines[idx+15].split()[2])
            instances[instancename]['minabsval'] = fracttofloat(outlines[idx+16].split()[4])
            instances[instancename]['maxabsval'] = fracttofloat(outlines[idx+17].split()[4])

        elif outline.startswith('Violation'):
            primviol = outlines[idx+2].split()[3]
            dualviol = outlines[idx+3].split()[3]
            if typeofvalue(primviol) in [int,float,fractions.Fraction] and typeofvalue(dualviol) in [int,float,fractions.Fraction]:
                instances[instancename]['pviol'] = float("{:.2e}".format(fracttofloat(primviol)))
                instances[instancename]['dviol'] = float("{:.2e}".format(fracttofloat(dualviol)))
            else:
                instances[instancename]['pviol'] = '-'
                instances[instancename]['dviol'] = '-'

        elif outline.startswith('Total time'):
            instances[instancename]['time'] = fracttofloat(outline.split()[3])
            instances[instancename]['readtime'] = fracttofloat(outlines[idx+1].split()[2])
            instances[instancename]['solvetime'] = fracttofloat(outlines[idx+2].split()[2])
            instances[instancename]['preproctime'] = fracttofloat(outlines[idx+3].split()[2])
            instances[instancename]['simplextime'] = fracttofloat(outlines[idx+4].split()[2])
            instances[instancename]['synctime'] = fracttofloat(outlines[idx+5].split()[2])
            instances[instancename]['transformtime'] = fracttofloat(outlines[idx+6].split()[2])
            instances[instancename]['rationaltime'] = fracttofloat(outlines[idx+7].split()[2])
            instances[instancename]['othertime'] = fracttofloat(outlines[idx+8].split()[2])

        elif outline.startswith('Refinements'):
            instances[instancename]['refs'] = int(outline.split()[2])
            instances[instancename]['stallrefs'] = int(outlines[idx+1].split()[2])
            instances[instancename]['pivrefs'] = int(outlines[idx+2].split()[2])
            instances[instancename]['feasrefs'] = int(outlines[idx+3].split()[2])
            instances[instancename]['unbdrefs'] = int(outlines[idx+4].split()[2])

        elif outline.startswith('Iterations'):
            instances[instancename]['iters'] = int(outline.split()[2])
            instances[instancename]['scratchiters'] = int(outlines[idx+1].split()[3])
            instances[instancename]['basisiters'] = int(outlines[idx+2].split()[3])
            instances[instancename]['primaliters'] = int(outlines[idx+3].split()[2])
            instances[instancename]['dualiters'] = int(outlines[idx+4].split()[2])
            instances[instancename]['flips'] = int(outlines[idx+5].split()[3])
            instances[instancename]['polish'] = int(outlines[idx+6].split()[-1])
            instances[instancename]['speed'] = round(fracttofloat(instances[instancename]['iters'])/max(instances[instancename]['solvetime'],tolerance),2)

        elif outline.startswith('LU factorizations'):
            instances[instancename]['lufacts'] = int(outline.split()[3])
            instances[instancename]['factortime'] = fracttofloat(outlines[idx+2].split()[3])

        elif outline.startswith('LU solves'):
            instances[instancename]['lusolves'] = int(outline.split()[3])
            instances[instancename]['lusolvetime'] = fracttofloat(outlines[idx+2].split()[3])

        elif decomp:
            if outline.startswith('Degeneracy'):
                instances[instancename]['ppiv'] = int(outlines[idx + 1].split()[3])
                instances[instancename]['dpiv'] = int(outlines[idx + 2].split()[3])
                instances[instancename]['primcand'] = int(outlines[idx + 3].split()[3])
                instances[instancename]['dualcand'] = int(outlines[idx + 4].split()[3])
                instances[instancename]['avgpdegen'] = fracttofloat(outlines[idx + 5].split()[3])
                instances[instancename]['avgddegen'] = fracttofloat(outlines[idx + 6].split()[3])

            elif outline.startswith('Algorithm Iterations'):
                instances[instancename]['algiter'] = int(outline.split()[2])

            elif outline.startswith('Decomp. Iterations'):
                instances[instancename]['rediter'] = int(outlines[idx + 3].split()[3])
                instances[instancename]['compiter'] = int(outlines[idx + 4].split()[3])

            elif outline.startswith('Red. Problem Status'):
                instances[instancename]['redprob'] = int(outline.split()[4])

            elif outline.startswith('Comp. Problem Status'):
                instances[instancename]['compprob'] = int(outline.split()[3])


# try parsing solution file
check_solu = False
try:
    with open(soluname):
        check_solu = True
except IOError:
    check_solu = False

if check_solu:
    solufile = open(soluname,'r')
    for soluline in solufile:
        solu = soluline.split()
        tag = solu[0]
        name = solu[1]
        if len(solu) == 3:
            value = solu[2]
            if typeofvalue(value) in [int,float]:
                value = fracttofloat(value)
        else:
            if tag == '=inf=':
                value = 'infeasible'
            else:
                value = 'unknown'
        if name in instances:
            instances[name]['soluval'] = value
            if not instances[name]['status'] in ['timeout', 'fail', 'abort']:
                # check solution status
                if value in ['infeasible', 'unbounded']:
                    if not instances[name]['status'] == value:
                        instances[name]['status'] = 'inconsistent'
                elif value == 'unknown':
                    instances[name]['status'] = 'not verified'
                elif (abs(instances[name]['value'] - value))/max(abs(instances[name]['value']),abs(value),tolerance) > tolerance:
                    instances[name]['status'] = 'inconsistent'
    solufile.close()

# save dictionary to file later use in compare script
with open(dataname, 'w') as f:
    json.dump(instances, f)

# count solution status
fails = sum(1 for name in instances if instances[name]['status'] == 'fail')
timeouts = sum(1 for name in instances if instances[name]['status'] == 'timeout')
infeasible = sum(1 for name in instances if instances[name]['status'] == 'infeasible')
unbounded = sum(1 for name in instances if instances[name]['status'] == 'unbounded')
optimal = sum(1 for name in instances if instances[name]['status'] == 'optimal')
aborts = sum(1 for name in instances if instances[name]['status'] == 'abort')
inconsistents = sum(1 for name in instances if instances[name]['status'] == 'inconsistent')

length = []
output = ''
# calculate maximum width of each column
for i,c in enumerate(columns):
    length.append(len(c))
    for name in instances:
        length[i] = max(length[i],len(str(instances[name].get(c,''))))
    if i == ncolumns:
        output = output + '  |'
    output = output + ' ' + c.rjust(length[i] + 1)

# print column header
print(output)
print('-'*len(output))

# print data for all instances with the computed length
for name in sorted(instances):
    output = ''
    for i,c in enumerate(columns):
        if i == ncolumns:
            output = output + '  |'
        output = output + ' ' + str(instances[name].get(c, '--')).rjust(length[i] + 1)
    print(output)

print('\nResults (testset '+testset+', settings '+settings+'):')
print('{} total: {} optimal, {} infeasible, {} unbounded, {} timeouts, {} inconsistents, {} fails, {} aborts'.format(len(instances),optimal,infeasible,unbounded,timeouts,inconsistents,fails,aborts))

# try to check for missing files
check_test = False
try:
    with open(testname):
        check_test = True
except IOError:
    print('No testset file found to check run for completeness.')

if not check_solu:
    print('No solution file found to check objective values.')

if check_test:
    testfile = open(testname,'r')
    printedMissing = False
    for testline in testfile:
        linesplit = testline.split('/')
        linesplit = linesplit[len(linesplit) - 1].rstrip(' \n').rstrip('.gz').rstrip('.GZ').rstrip('.z').rstrip('.Z')
        linesplit = linesplit.split('.')
        instancename = linesplit[0]
        for i in range(1, len(linesplit)-1):
            instancename = instancename + '.' + linesplit[i]
        if not instancename in instances:
            if not printedMissing:
                print('\n')
            print('missing instance: '+instancename)
            printedMissing = True

    testfile.close()
