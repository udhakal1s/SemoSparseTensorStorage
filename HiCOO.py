"""
Variables initialized to track a number of elements needed to perform calculations
"""

ibitsadded = 0
jbitsadded = 0
kbitsadded = 0
ielements = [0]
jelements = [0]
kelements = [0]
elementcount = 0
bitcount = 0
iadder = 0x01
jadder = 0x1
kadder = 0x1
iaddcount = 0
jaddcount = 0
kaddcount = 0


iblock = []
jblock = []
kblock = []
block = []
lastiblock = 0
lastjblock = 0
lastkblock = 0
blockholding = []
icount = 0
jcount = 0
kcount = 0
bptr = []
bptrcount = 0

values = []

superblock = []


"""
this function reads the input file and sends the numbers to their respective locations
"""


def fileToHiCOO(file):
    tempNum = ''
    with open(file) as f:
        for line in f:
            count = 0
            indcount = 0
            mode = getMode(line)
            for x in line:
                if count != len(line)-1:
                    if x.isnumeric() or x == '.':
                        tempNum += x
                    elif mode != indcount:
                        indcount += 1
                        indicebuffer(int(tempNum), indcount)
                        tempNum = ''
                    else:
                        addtovalues(float(tempNum))
                        tempNum = ''
                    count += 1
                else:
                    if x.isnumeric():
                        tempNum += x
                        addtovalues(float(tempNum))
                        tempNum = ''
                    elif tempNum != '' and tempNum != ' ':
                        addtovalues(float(tempNum))
                        tempNum = ''


"""
this function is called by filetoHiCOO function with all values to be stored in an array
"""


def addtovalues(floatnum):
    values.append(floatnum)


""""
#This function below is called from the indicebuffer funciton to determine if block indices are 
able to be stored in the current block or to create a new one
"""


def proxi():
    iclose = abs(blockholding[0] - lastiblock)
    jclose = abs(blockholding[1] - lastjblock)
    kclose = abs(blockholding[2] - lastkblock)
    if  iclose < 2 and  jclose < 2 and  kclose < 2:
        if lastiblock % 2 != 0 and lastjblock % 2 != 0 and lastkblock % 2 != 0:
            return True
        else:
            return False
    else:
        return False


"""
This function below takes coordinates from main HiCOO funciton and waits until all coordinates of each value are recieved.
Using these coordinates, it determines if they can be stored in the same block or not
"""

def indicebuffer(indice, num):
    blockind = evenorodd(indice, num)
    addtoelement(indice % 2, num)

    if num < 3:
        blockholding.append(blockind)
    else:
        blockholding.append(blockind)
        proximity = proxi()
        addtoindices(proximity)


"""
This function takes a coordinate and returns the coordinate in the format needed for bi, bj, bk values
"""


def evenorodd(indice, num):
    blockind = 0
    evenodd = indice % 2
    if evenodd == 0 and indice != 1 and indice != 0:
        blockind = (indice - evenodd) / 2
    elif indice == 1:
        blockind = indice -1
    elif indice == 0:
        blockind = indice
    return int(blockind)


"""
This function determines if the values can be stored in the same block, or if a new one is needed, then adds the values to
the resulting decision
"""


def addtoindices(proximity):
    global lastiblock
    global lastjblock
    global lastkblock
    global iblock
    global jblock
    global kblock
    global bptrcount
    if proximity:
        blockholding.clear()
        bptrcount += 1
    else:
        iblock.append(blockholding[0])
        lastiblock = blockholding[0]
        jblock.append(blockholding[1])
        lastjblock = blockholding[1]
        kblock.append(blockholding[2])
        lastkblock = blockholding[2]
        bptrcount += 1
        bptr.append(bptrcount)
        blockholding.clear()


"""
This function takes the number needed to be stored for storage saving,
It does so using some bitwise operations
"""


def addtoelement(binnum, num):
    global icount
    global ibitsadded
    global iadder
    global jcount
    global jbitsadded
    global jadder
    global kcount
    global kbitsadded
    global kadder
    global iaddcount
    global jaddcount
    global kaddcount

    #this if statement determines if the number returned is a 1 or 0 to be storaed in binary
    if binnum == 1:
        if num == 1:
            temp = ielements[icount]

            temp = temp | iadder
            ielements[icount] = temp
            iadder <<= 1
            iaddcount += 1
            ibitsadded += 1
            if ibitsadded == 8:
                iadder >>= 8
                ibitsadded -= 8
                ielements.append(0)
                icount += 1
        elif num == 2:
            temp = jelements[jcount]
            temp = temp | jadder
            jelements[jcount] = temp
            jadder <<= 1
            jaddcount += 1
            jbitsadded += 1
            if jbitsadded == 8:
                jadder >>= 8
                jbitsadded -= 8
                jcount += 1
                jelements.append(0)
        else:
            temp = kelements[kcount]
            temp = temp | kadder
            kelements[kcount] = temp
            kadder <<= 1
            kaddcount += 1
            kbitsadded += 1
            if kbitsadded == 8:
                kadder >>= 8
                kbitsadded -= 8
                kcount += 1
                kelements.append(0)
    else:
        if num == 1:
            temp = ielements[icount]
            ielements[icount] = temp
            iadder <<= 1
            ibitsadded += 1
            if ibitsadded == 8:
                iadder >>= 8
                ibitsadded -= 8
                ielements.append(0)
                icount += 1
        elif num == 2:
            temp = jelements[jcount]
            jelements[jcount] = temp
            jadder <<= 1
            jbitsadded += 1
            if jbitsadded == 8:
                jadder >>= 8
                jbitsadded -= 8
                jcount += 1
                jelements.append(0)
        else:
            temp = kelements[kcount]
            kelements[kcount] = temp
            kadder <<= 1
            kbitsadded += 1
            if kbitsadded == 8:
                kadder >>= 8
                kbitsadded -= 8
                kcount += 1
                kelements.append(0)


"""
This functions gets the mode for an optimistic use to make the function non-mode specific
"""


def getMode(line):
    tempNum = ''
    arrayholder = []
    for x in line:
        if x.isnumeric() or x == '.':
            tempNum += x
        elif tempNum != '' and tempNum != ' ':
            arrayholder.append(float(tempNum))
            tempNum = ''
    if tempNum != '' and tempNum != ' ':
        arrayholder.append(float(tempNum))
    return len(arrayholder) - 1


"""
This function sends all arrays to the "superblock", but due to difficulty manly just stores the correctly formatted entries.
"""


def tosuperblock():
    global superblock
    superblock = [bptr, iblock, jblock, kblock, ielements, jelements, kelements, values]

"""
This function prints the "superblock" line by line
"""


def printsuperblock():
    for i in superblock:
        print(i)

