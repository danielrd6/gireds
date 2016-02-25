def check(sci_file):

    f = open(sci_file, 'r')
    a = f.read()
    d = eval(a)
    f.close()

    for i in d:
        for j in i:
            if i[j] == '':
                print(
                    'Image {:s} of object {:s} is missing the {:s} exposure.'
                    .format(i['image'], i['object'], j))

if __name__ == '__main__':
    sci_file = 'file_associations_sci.dat'
    check(sci_file)
