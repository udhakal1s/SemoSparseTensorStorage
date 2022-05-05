# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # print(round(x))
    HiCOO.fileToHiCOO('venv/tensor1.tn')
    print('ei array = ', HiCOO.ielements)
    print('ej array = ', HiCOO.jelements)
    print('ek array = ', HiCOO.kelements)
    print('bi array = ', HiCOO.iblock)
    print('bj array = ', HiCOO.jblock)
    print('bk array = ', HiCOO.kblock)
    print('Values array = ', HiCOO.values)
    print('Block Pointer array = ', HiCOO.bptr)
    HiCOO.tosuperblock()
    print()
    HiCOO.printsuperblock()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
